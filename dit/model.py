from omegaconf import OmegaConf
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from sat.model.base_model import BaseModel
from sat.model.transformer import CrossAttention
from sat.model.mixins import BaseMixin
from sat.helpers import print_rank0
from sat.transformer_defaults import standard_attention

from sat.ops.layernorm import LayerNorm

from dit.TaylorSeer.taylor_wrapper import TaylorSeerWrapper
from dit.embeddings import TimeEmbedding, ConditionEmbedding, ImagePatchEmbeddingMixin, BasicPositionEmbeddingMixin
from dit.embeddings import DDPMTimeEmbedding, RotaryPositionEmbedding

from dit.utils import instantiate_from_config, append_dims
from sat.mpu.utils import split_tensor_along_last_dim
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from sat.mpu.utils import unscaled_init_method


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def unpatchify(x, c, p, rope_position_ids=None, hw=None):
    """
    x: (N, T, patch_size**2 * C)
    imgs: (N, H, W, C)
    """
    if False:
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum('nlpqc->ncplq', x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        if hw is None:
            h = w = int(x.shape[1] ** 0.5)
        else:
            h, w = hw
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
    
    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(self, hidden_size, patch_size, num_patches, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def final_forward(self, logits, **kwargs):
        x, emb = logits, kwargs['emb']
        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None), hw=kwargs.get('hw', None))

    def reinit(self, parent_model=None):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class SimpleFinalLayerMixin(BaseMixin):
    def __init__(self, hidden_size, patch_size, num_patches, out_channels):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def final_forward(self, logits, **kwargs):
        x = self.linear(logits)
        return unpatchify(x, c=self.out_channels, p=self.patch_size, rope_position_ids=kwargs.get('rope_position_ids', None))

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

class AdaLNMixin(BaseMixin):
    def __init__(self, hidden_size, num_layers, image_size, pos_embed_config, is_decoder=False,
                 nogate=False, cross_adaln=False, use_block_attention=False,
                 num_patches=None, qk_ln=False, num_head=None, random_position=False, block_size=None,
                 re_position=False, cross_lr=False, args=None, params_dtype=torch.float32):
        super().__init__()
        self.nogate = nogate
        self.cross_adaln = cross_adaln
        self.image_size = image_size
        self.num_patches = num_patches
        self.block_size = block_size
        self.sr_scale = args.sr_scale
        self.patch_size = args.patch_size
        if nogate:
            out_times = 4
        else:
            out_times = 6
        if cross_adaln:
            out_times = (out_times // 2) * 3

        self.qk_ln = qk_ln
        hidden_size_head = hidden_size // num_head
        if qk_ln:
            print("--------use qk_ln--------")
            self.q_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])
            self.k_layer_norm = nn.ModuleList([
                nn.LayerNorm(hidden_size_head, eps=1e-6)
                for _ in range(num_layers)
            ])

        self.adaLN_modulations = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, out_times * hidden_size)
            ) for _ in range(num_layers)
            ])
        self.is_decoder = is_decoder
        self.use_block_attention = use_block_attention
        if random_position:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 8, hidden_size=hidden_size, hidden_size_head=hidden_size//num_head)
        else:
            self.rope = RotaryPositionEmbedding(pix2struct=True, num_patches=image_size * 2, hidden_size=hidden_size,
                                                hidden_size_head=hidden_size // num_head)
        self.re_position = re_position

        if re_position:
            self.re_pos_embed = nn.Parameter(torch.zeros(num_layers, 4, num_head, hidden_size_head))

        self.cross_lr = cross_lr
        if cross_lr:
            in_channels = 3 # latent is 4
            self.lr_patch_size = 2
            bias = True
            self.proj_lr = nn.Conv2d(in_channels, hidden_size, kernel_size=self.lr_patch_size, stride=self.lr_patch_size, bias=bias)
            if hasattr(args, 'fp16') and args.fp16:
                params_dtype = torch.half
            elif hasattr(args, 'bf16') and args.bf16:
                params_dtype = torch.bfloat16
            else:
                params_dtype = torch.float32
            self.cross_attention = CrossAttention(
                hidden_size,
                args.num_attention_heads,
                0,
                0,
                unscaled_init_method(0.02),
                0,
                hidden_size_per_attention_head=None,
                output_layer_init_method=unscaled_init_method(0.02),
                cross_attn_hidden_size=None,
                bias=True,
                hooks=None,
                transformer_pointer=self,
                params_dtype=params_dtype
            )
            self.post_cross_attention_layernorm = LayerNorm(hidden_size, eps=1e-6)
            head = args.num_attention_heads
            hidden_size_per_attention_head = hidden_size // head
            self.lr_query_position_embedding = nn.Parameter(torch.zeros(1, self.block_size ** 2, head, hidden_size_per_attention_head))
            self.lr_block_size = self.block_size * self.patch_size // self.sr_scale // self.lr_patch_size
            self.lr_key_position_embedding = nn.Parameter(torch.zeros(1, (self.lr_block_size * 3) ** 2, head, hidden_size_per_attention_head))
            torch.nn.init.normal_(self.lr_query_position_embedding, std=0.02)
            torch.nn.init.normal_(self.lr_key_position_embedding, std=0.02)
            if self.qk_ln:
                self.cross_q_ln = nn.LayerNorm(hidden_size_head, eps=1e-6)
                self.cross_k_ln = nn.LayerNorm(hidden_size_head, eps=1e-6)

    def position_embedding_forward(self, position_ids, **kwargs):
        return None

    def attention_forward(self, hidden_states, mask, rope_position_ids, inference=0, direction="lt", mems=None, do_concat=True, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention

        mixed_raw_layer = self.query_key_value(hidden_states)
        (query_layer,
         key_layer,
         value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        # [b, h, n*n, d/h]
        h, w = kw_args['hw']
        if origin.use_block_attention:
            mask = None
            block_size = origin.block_size
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            in_x = h // block_size
            in_y = w // block_size
            query_layer = query_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            key_layer = key_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            value_layer = value_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)
            if origin.qk_ln:
                query_layernorm = origin.q_layer_norm[kw_args['layer_id']]
                key_layernorm = origin.q_layer_norm[kw_args['layer_id']]
                query_layer = query_layernorm(query_layer)
                key_layer = key_layernorm(key_layer)

            if origin.rope:
                query_layer = origin.rope(query_layer, rope_position_ids=rope_position_ids)
                key_layer = origin.rope(key_layer, rope_position_ids=rope_position_ids)

            def transform(x):
                x = rearrange(x, 'b (n m) h d -> b n m h d', n=h, m=w)
                x = rearrange(x, 'b (x l) (y w) h d -> b x y (l w) h d', l=block_size, w=block_size)
                return x

            def inverse_transform(x):
                x = rearrange(x, '(b x y) (l w) h d -> b (x l y w) (h d)', l=block_size, w=block_size, x=in_x, y=in_y)
                return x

            def get_cat_layer_left_top(x):
                cat_x = torch.cat([x[:, 0:1], x], dim=1)
                cat_x = torch.cat([cat_x[:, :, 0:1], cat_x], dim=2)
                x = torch.cat([x, cat_x[:, :x.shape[1], :x.shape[2]],
                               cat_x[:, 1:, :x.shape[2]], cat_x[:, :x.shape[1], 1:]], dim=3)
                #self lef-top left top
                return x

            def get_cat_layer_right_bottom(x):
                cat_x = torch.cat([x, x[:, -1:]], dim=1)
                cat_x = torch.cat([cat_x, cat_x[:, :, -1:]], dim=2)
                x = torch.cat([x, cat_x[:, 1:, 1:], cat_x[:, 1:, :x.shape[2]],
                               cat_x[:, :x.shape[1], 1:]], dim=3)
                return x

            def get_cat_layer_left_top_withcache(x, mems, index):
                if mems[2] is not None: # top
                    cat_x = torch.cat([mems[2][kw_args['layer_id']]['mem_kv'][index], x], dim=1)
                else:
                    cat_x = torch.cat([x[:, 0:1], x], dim=1)
                if mems[0] is not None: #left
                    if mems[1] is not None: # left-top
                        left_cat = torch.cat([mems[1][kw_args['layer_id']]['mem_kv'][index][:, :, -1:], mems[0][kw_args['layer_id']]['mem_kv'][index+2]], dim=1)
                    else:
                        left_cat = torch.cat([mems[0][kw_args['layer_id']]['mem_kv'][index+2][:, 0:1], mems[0][kw_args['layer_id']]['mem_kv'][index+2]], dim=1)
                    cat_x = torch.cat([left_cat, cat_x], dim=2)
                else:
                    cat_x = torch.cat([cat_x[:, :, 0:1], cat_x], dim=2)
                x = torch.cat([x, cat_x[:, :x.shape[1], :x.shape[2]],
                               cat_x[:, 1:, :x.shape[2]], cat_x[:, :x.shape[1], 1:]], dim=3)
                return x

            query_layer = transform(query_layer)
            key_layer = transform(key_layer)
            value_layer = transform(value_layer)

            if not do_concat:
                pass
            elif inference == 1:
                kw_args['output_this_layer']['mem_kv'] = [key_layer, value_layer]
                k_stack = [key_layer]
                v_stack = [value_layer]
                for mem in mems:
                    k_stack.append(mem[kw_args['layer_id']]['mem_kv'][0])
                    v_stack.append(mem[kw_args['layer_id']]['mem_kv'][1])
                key_layer = torch.cat(k_stack, dim=3)
                value_layer = torch.cat(v_stack, dim=3)
            elif inference == 2:
                # mems:left, left-top, top
                kw_args['output_this_layer']['mem_kv'] = [key_layer[:, -1:].clone(), value_layer[:, -1:].clone(), key_layer[:, :, -1:].clone(), value_layer[:, :, -1:].clone()]
                key_layer = get_cat_layer_left_top_withcache(key_layer, mems, 0)
                value_layer = get_cat_layer_left_top_withcache(value_layer, mems, 1)
            elif direction == "lt":
                key_layer = get_cat_layer_left_top(key_layer)
                value_layer = get_cat_layer_left_top(value_layer)
            elif direction == "rb":
                key_layer = get_cat_layer_right_bottom(key_layer)
                value_layer = get_cat_layer_right_bottom(value_layer)
            else:
                raise NotImplementedError

            query_layer = query_layer.view(-1, *query_layer.shape[3:])
            key_layer = key_layer.view(-1, *key_layer.shape[3:])
            value_layer = value_layer.view(-1, *value_layer.shape[3:])

            if origin.re_position and do_concat:
                re_pos_embed = origin.re_pos_embed[kw_args['layer_id']].repeat_interleave(key_layer.shape[1] // origin.re_pos_embed.shape[1], dim=0).unsqueeze(0)
                key_layer = key_layer + re_pos_embed

            context_layer = flash_attn_func(query_layer, key_layer, value_layer)

            context_layer = inverse_transform(context_layer)
        else:
            query_layer = self._transpose_for_scores(query_layer)
            key_layer = self._transpose_for_scores(key_layer)
            value_layer = self._transpose_for_scores(value_layer)
            context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output

    def process_lr(self, lr_imgs):
        lr_imgs = self.proj_lr(lr_imgs)
        lr_hidden_size = lr_imgs.shape[1]

        unFold = torch.nn.Unfold(kernel_size=3 * self.lr_block_size, stride=self.lr_block_size,
                                 padding=self.lr_block_size)
        lr_imgs = unFold(lr_imgs)

        lr_imgs = lr_imgs.view(lr_imgs.shape[0], lr_hidden_size, self.lr_block_size * 3, self.lr_block_size * 3, -1)
        lr_imgs = lr_imgs.permute(0, 4, 2, 3, 1).contiguous()
        lr_imgs = lr_imgs.view(lr_imgs.shape[0] * lr_imgs.shape[1], -1, lr_imgs.shape[-1])
        return lr_imgs

    def cross_attention_forward(self, hidden_states, lr_imgs, **kw_args):
        h, w = kw_args['hw']
        block_size = self.block_size
        in_x = h // block_size
        in_y = w // block_size
        def transform(x):
            x = rearrange(x, 'b (n m) h d -> b n m h d', n=h, m=w)
            x = rearrange(x, 'b (x l) (y w) h d -> b x y (l w) h d', l=block_size, w=block_size)
            return x

        def inverse_transform(x):
            x = rearrange(x, '(b x y) (l w) h d -> b (x l y w) (h d)', l=block_size, w=block_size, x=in_x, y=in_y)
            return x

        origin = self
        self = self.cross_attention
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        query_layer = self.query(hidden_states)
        mixed_x_layer = self.key_value(lr_imgs)
        key_layer, value_layer = split_tensor_along_last_dim(mixed_x_layer, 2)
        query_layer = query_layer.view(batch_size, seq_len, -1, self.hidden_size_per_attention_head)

        key_layer = key_layer.view(key_layer.shape[0], lr_imgs.shape[1], -1, self.hidden_size_per_attention_head)
        value_layer = value_layer.view(key_layer.shape[0], lr_imgs.shape[1], -1, self.hidden_size_per_attention_head)


        query_layer = transform(query_layer)
        query_layer = query_layer.view(-1, *query_layer.shape[3:])

        if origin.qk_ln:
            query_layer = origin.cross_q_ln(query_layer)
            key_layer = origin.cross_k_ln(key_layer)

        query_layer = query_layer + origin.lr_query_position_embedding
        key_layer = key_layer + origin.lr_key_position_embedding

        context_layer = flash_attn_func(query_layer, key_layer, value_layer)
        context_layer = inverse_transform(context_layer)
        # Output. [b, s, h]
        output = self.dense(context_layer)
        if self.training:
            output = self.output_dropout(output)
        return output

    def layer_forward(self, hidden_states, mask, do_concat=True, *args, **kwargs):
        layer_id = kwargs['layer_id']
        layer = self.transformer.layers[kwargs['layer_id']]
        adaLN_modulation = self.adaLN_modulations[kwargs['layer_id']]
        if self.nogate and self.cross_adaln:
            shift_msa, scale_msa, shift_cross, scale_cross, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa = gate_cross = gate_mlp = 1
        elif self.nogate and not self.cross_adaln:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(kwargs['emb']).chunk(4, dim=1)
            gate_msa = gate_mlp = 1
        elif not self.nogate and self.cross_adaln:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(9, dim=1)
            gate_msa, gate_cross, gate_mlp = gate_msa.unsqueeze(1), gate_cross.unsqueeze(1), gate_mlp.unsqueeze(1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(kwargs['emb']).chunk(6, dim=1)
            gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)

        # self attention
        attention_input = layer.input_layernorm(hidden_states)
        attention_input = modulate(attention_input, shift_msa, scale_msa)

        concat = kwargs.get('concat', None)
        if concat is not None and concat[0] != concat[-1]:
            input1, input2 = attention_input.chunk(2)
            rope_position_ids1, rope_position_ids2 = kwargs['rope_position_ids'].chunk(2)
            kwargs['rope_position_ids'] = rope_position_ids1
            output1 = layer.attention(input1, mask, do_concat=concat[0], **kwargs)
            kwargs['rope_position_ids'] = rope_position_ids2
            output2 = layer.attention(input2, mask, do_concat=concat[-1], **kwargs)
            attention_output = torch.cat([output1, output2], dim=0)
        else:
            attention_output = layer.attention(attention_input, mask, do_concat=do_concat, **kwargs)

        if self.transformer.layernorm_order == 'sandwich':
            attention_output = layer.third_layernorm(attention_output)
        hidden_states = hidden_states + gate_msa * attention_output
        if layer_id == 0 and self.cross_lr:
            cross_attention_input = layer.post_attention_layernorm(hidden_states)

            # do cross attention here
            cross_attention_output = self.cross_attention_forward(cross_attention_input, **kwargs)

            hidden_states = hidden_states + cross_attention_output
            mlp_input = self.post_cross_attention_layernorm(hidden_states)
        else:
            mlp_input = layer.post_attention_layernorm(hidden_states)

        mlp_input = modulate(mlp_input, shift_mlp, scale_mlp)
        mlp_output = layer.mlp(mlp_input, **kwargs)
        if self.transformer.layernorm_order == 'sandwich':
            mlp_output = layer.fourth_layernorm(mlp_output)
        hidden_states = hidden_states + gate_mlp * mlp_output
        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)
class DiffusionEngine(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kwargs):
        self.image_size = args.image_size
        self.patch_size = args.patch_size
        self.num_patches = (args.image_size // args.patch_size)**2
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.scale_factor = args.scale_factor
        self.input_time = args.input_time
        self.is_decoder = args.is_decoder
        self.no_crossmask = args.no_crossmask
        self.stop_grad_patch_embed = args.stop_grad_patch_embed
        self.sr_scale = args.sr_scale
        self.random_position = args.random_position
        self.random_direction = args.random_direction
        self.image_block_size = args.image_block_size
        if self.random_direction:
            print("--------use random direction--------")
        if self.random_position:
            print("--------use random position--------")

        self.use_block_attention = True

        self.use_taylorseer = getattr(args, 'use_taylorseer', False)
        self.taylorseer = None

        if self.use_taylorseer:
            self.taylorseer = TaylorSeerWrapper(
                self,
                interval=args.taylor_interval,
                max_order=args.taylor_order,
                test_flops=args.test_flops
            )

        if 'activation_func' not in kwargs:
            approx_gelu = nn.GELU(approximate='tanh')
            kwargs['activation_func'] = approx_gelu
        args.parallel_output = parallel_output
        super().__init__(args, transformer=transformer, layernorm=partial(LayerNorm, elementwise_affine=False, eps=1e-6),  **kwargs)
        
        configs = OmegaConf.load(args.config_path)
        module_configs = configs.pop('modules', None)
        modeling_configs = configs.pop('modeling', None)


        self._build_modules(args, module_configs)
        self._build_modeling(args, modeling_configs)

        self.collect_attention = None
    def _build_modules(self, args, module_configs):

        pos_embed_config = module_configs.pop('position_embedding_config')
        if self.input_time == "adaln":
            self.add_mixin('adaln_layer', AdaLNMixin(args.hidden_size, args.num_layers, is_decoder=args.is_decoder,
                                                     image_size=args.image_size//args.patch_size,
                                                     nogate=args.nogate, cross_adaln=args.cross_adaln,
                                                     use_block_attention=self.use_block_attention, pos_embed_config=pos_embed_config,
                                                     num_patches=self.num_patches, num_head=args.num_attention_heads,
                                                     qk_ln=args.qk_ln, random_position=args.random_position,
                                                     block_size=args.image_block_size//args.patch_size,
                                                     re_position=args.re_position,
                                                     cross_lr=args.cross_lr, args=args), reinit=True)
            self.add_mixin('final_layer', FinalLayerMixin(args.hidden_size, args.patch_size, self.num_patches, args.out_channels), reinit=True)
        elif self.input_time in ["concat", "add"]:
            self.add_mixin('final_layer', SimpleFinalLayerMixin(args.hidden_size, args.patch_size, self.num_patches, args.out_channels), reinit=True)
        else:
            raise NotImplementedError

        append_emb = self.input_time == "concat"
        add_emb = self.input_time == "add"

        self.cross_lr = args.cross_lr
        self.add_mixin('patch_embed', ImagePatchEmbeddingMixin(args.in_channels, args.hidden_size, args.patch_size, append_emb=append_emb, add_emb=add_emb, reg_token_num=args.reg_token_num), reinit=True)


        scale_factor = module_configs.pop('scale_factor', 1)
        self.scale_factor = scale_factor
        first_stage_config = module_configs.pop('first_stage_config', None)
        if first_stage_config:
            self.first_stage_model = instantiate_from_config(first_stage_config)
            self._init_first_stage()
        else:
            self.first_stage_model = None

        if args.ddpm_time_emb:
            self.time_embed = DDPMTimeEmbedding(args.hidden_size)
        else:
            self.time_embed = TimeEmbedding(args.hidden_size)
        self.cond_embed = ConditionEmbedding(args.hidden_size, args.label_dim, args.augment_dim, args.vector_dim, args.label_dropout)
        
        if args.is_decoder:
            text_encoder_config = module_configs.pop('text_encoder_config')
            self.text_encoder = instantiate_from_config(text_encoder_config)
        else:
            self.text_encoder = None

        if args.image_condition:
            image_encoder_config = module_configs.pop('image_encoder_config')
            self.image_encoder = instantiate_from_config(image_encoder_config)
        else:
            self.image_encoder = None

    def _build_modeling(self, args, modeling_configs):
        precond_config = modeling_configs.pop('precond_config')
        self.precond = instantiate_from_config(precond_config)

        loss_config = modeling_configs.pop('loss_config')
        self.loss_func = instantiate_from_config(loss_config)
        sampler_config = modeling_configs.pop('sampler_config')
        self.sampler = instantiate_from_config(sampler_config)
        print(f"[DEBUG] sampler class: {type(self.sampler)}")

        
    
    def disable_untrainable_params(self):
        disable_prefixs = ["text_encoder", "first_stage_model", "image_encoder"]
        total_trainable = 0
        for n, p in self.named_parameters():
            flag = False
            for prefix in disable_prefixs:
                if n.startswith(prefix):
                    flag = True
                    break
            if flag:
                p.requires_grad_(False)
            else:
                total_trainable += p.numel()
        if self.stop_grad_patch_embed:
            self.mixins.patch_embed.proj.weight.requires_grad_(False)
            self.mixins.patch_embed.proj.bias.requires_grad_(False)
            total_trainable -= self.mixins.patch_embed.proj.weight.numel()
            total_trainable -= self.mixins.patch_embed.proj.bias.numel()

        print_rank0("***** Total trainable parameters: "+str(total_trainable)+" *****")
                
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Diffusion', 'Diffusion Configurations')
        group.add_argument('--image-size', type=int, default=64)
        group.add_argument('--patch-size', type=int, default=4)
        group.add_argument('--in-channels', type=int, default=3)
        group.add_argument('--out-channels', type=int, default=3)
        group.add_argument('--scale-factor', type=float, default=1)
        group.add_argument('--cross-attn-hidden-size', type=int, default=640)
        group.add_argument('--augment-dim', type=int, default=0)
        group.add_argument('--label-dim', type=int, default=0)
        group.add_argument('--label-dropout', type=float, default=0)
        group.add_argument('--is-decoder', action='store_true')
        group.add_argument('--input-time', type=str, choices=["adaln", "concat", "concatv2", "add"], default="adaln")
        group.add_argument('--ddpm-time-emb', action='store_true')
        group.add_argument('--reg-token-num', type=int, default=0)
        group.add_argument('--nogate', action='store_true')
        group.add_argument('--cross-adaln', action='store_true')
        group.add_argument('--no-crossmask', action='store_true')
        group.add_argument('--stop-grad-patch-embed', action='store_true')
        group.add_argument('--text-dropout', type=float, default=0)
        group.add_argument('--config-path', type=str, default=None)
        group.add_argument('--sr-scale', type=int, default=4)
        group.add_argument('--qk-ln', action='store_true')
        group.add_argument('--random-position', action='store_true')
        group.add_argument('--random-direction', action='store_true')
        group.add_argument('--image-block-size', type=int, default=128)
        group.add_argument('--vector-dim', type=int, default=0)
        group.add_argument('--image-condition', action='store_true')
        group.add_argument('--lr-dropout', default=0, type=float)
        group.add_argument('--re-position', action='store_true')
        group.add_argument('--cross-lr', action='store_true')
        # === TaylorSeer 参数 ===
        group.add_argument('--use-taylorseer', action='store_true',
                           help='Enable TaylorSeer inference skipping')
        group.add_argument('--taylor-interval', type=int, default=4,
                           help='Sampling interval for Taylor expansion')
        group.add_argument('--taylor-order', type=int, choices=[1, 2], default=2,
                           help='Order of Taylor expansion (1 or 2)')
        group.add_argument('--test-flops', action='store_true',
                           help='Print estimated FLOPs statistics')

        return parser
    
    @classmethod
    def get_data_config(cls, args):
        configs = OmegaConf.load(args.config_path)
        data_config = configs.get('data', {})
        return data_config
        
    def model_forward(self, *args, **kwargs):
        sigmas = kwargs["sigmas"]
        emb = self.time_embed(sigmas)
        emb = self.cond_embed(emb, **kwargs)
        emb = F.silu(emb)
        kwargs['emb'] = emb
        
        kwargs['input_ids'] = kwargs['position_ids'] = kwargs['attention_mask'] = torch.ones((1,1)).to(sigmas.dtype)
        return super().forward(*args, **kwargs)

    def precond_forward(self, inference, rope_position_ids, concat_lr_imgs, lr_imgs=None, ar=False,
                        ar2=False, sample_step=None, block_batch=1, *args, **kwargs):
        """
        执行扩散模型的前向推理，支持三种推理模式：全图推理、逐块自回归推理(ar)和批量块自回归推理(ar2)

        参数:
            inference: 布尔值，指示是否处于推理模式
            rope_position_ids: 用于旋转位置编码(RoPE)的位置ID张量
            concat_lr_imgs: 用于与主图像拼接的低分辨率图像
            lr_imgs: 原始低分辨率图像，用于条件生成，可选
            ar: 布尔值，是否使用自回归模式(每次处理一个块)
            ar2: 布尔值，是否使用批量自回归模式(每次处理多个块)
            sample_step: 当前采样步骤，用于确定处理方向
            block_batch: 在ar2模式下每次处理的块批次大小
            *args, **kwargs: 传递给model_forward的额外参数

        返回:
            处理后的图像张量
        """
        # 从kwargs中获取输入图像和噪声水平
        images, sigmas = kwargs["images"], kwargs["sigmas"]

        # 将低分辨率图像与主图像在通道维度上拼接，增加条件信息
        images = torch.cat((images, concat_lr_imgs), dim=1)

        # TODO: 优化低分辨率图像的内存使用
        # lr_imgs = lr_imgs[:, :, :128, :128]

        # 如果启用了cross_lr选项，对低分辨率图像进行预处理
        if self.mixins['adaln_layer'].cross_lr:
            lr_imgs = self.mixins['adaln_layer'].process_lr(lr_imgs)

        # 获取图像的高度和宽度
        h, w = images.shape[2:4]

        # 计算扩散过程中的各种系数
        # c_skip: 原始图像的权重系数
        # c_out: 模型输出的权重系数
        # c_in: 模型输入的缩放系数
        # c_noise: 噪声的缩放系数
        c_skip, c_out, c_in, c_noise = map(lambda t: t.to(images.dtype), self.precond(append_dims(sigmas, images.ndim)))

        # 应用缩放因子调整图像值范围
        images *= self.scale_factor

        # 模式1：自回归(AR)模式 - 每次处理一个块
        if inference and ar:  # block_batch=1
            # 确保block_batch为1，因为AR模式下每次只处理一个块
            assert block_batch == 1

            # 设置块的大小和ViT(Vision Transformer)块大小
            block_size = self.image_block_size  # 图像块大小(像素)
            vit_block_size = block_size // self.patch_size  # ViT块大小(patch数量)

            # 计算块的行数和列数
            block_h, block_w = h // block_size, w // block_size

            # 重塑位置ID张量以匹配图像块的结构
            rope_position_ids = rope_position_ids.view(-1, h // self.patch_size, w // self.patch_size, 2)

            samples = []  # 存储每一行的处理结果
            cached = [None] * block_w  # 缓存每列()的处理结果

            # 根据随机方向选项和当前采样步骤确定处理方向
            # 偶数步骤从左上到右下，奇数步骤从右下到左上(如果启用了random_direction)
            if self.random_direction and sample_step is not None and sample_step % 2 == 1:
                # 从右下到左上的处理顺序
                range_i = range(block_h - 1, -1, -1)
                range_j = range(block_w - 1, -1, -1)
            else:
                # 从左上到右下的处理顺序
                range_i = range(block_h)
                range_j = range(block_w)

            # 按行遍历图像块
            for i in range_i:
                previous = None  # 存储上一个块的处理结果
                sample_row = []  # 存储当前行的处理结果

                # 按列遍历图像块
                for j in range_j:
                    # 提取当前块的图像数据
                    tmp_images = images[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

                    # 提取当前块的位置ID
                    tmp_position_ids = rope_position_ids[:, i * vit_block_size:(i + 1) * vit_block_size,
                                       j * vit_block_size:(j + 1) * vit_block_size].contiguous().view(-1,
                                                                                                      vit_block_size * vit_block_size,
                                                                                                      2)

                    # 更新kwargs中的参数
                    kwargs["images"] = tmp_images * c_in  # 应用输入缩放系数
                    kwargs["sigmas"] = c_noise.reshape(-1)  # 重塑噪声系数
                    kwargs["rope_position_ids"] = tmp_position_ids  # 更新位置ID

                    mems = []  # 用于存储记忆值的列表

                    # 如果当前列有缓存的记忆值，加入mems
                    if cached[j] is not None:
                        mems.append(cached[j])

                    # 如果不是第一列，使用前一列的缓存和前一个块的结果作为条件
                    if j != 0:
                        if cached[j - 1] is not None:
                            mems.append(cached[j - 1])
                        mems.append(previous)

                    # 计算当前块对应的低分辨率图像索引
                    lr_id = i * block_w + j

                    # 执行模型前向传播
                    output, *output_per_layers = self.model_forward(*args, hw=[vit_block_size, vit_block_size],
                                                                    mems=mems, inference=1,
                                                                    lr_imgs=lr_imgs[lr_id:lr_id + 1], **kwargs)

                    # 应用输出缩放系数和跳跃连接
                    output = output * c_out + tmp_images[:, :self.out_channels] * c_skip

                    # 更新缓存
                    if j != 0:
                        cached[j - 1] = previous

                    # 根据位置更新缓存或previous变量
                    if j == block_w - 1:  # 如果是行的最后一个块
                        cached[j] = output_per_layers
                    else:
                        previous = output_per_layers

                    # 将处理后的块加入当前行结果
                    sample_row.append(output)

                # 在水平方向上拼接当前行的所有块
                sample_row = torch.cat(sample_row, dim=3)
                samples.append(sample_row)

            # 在垂直方向上拼接所有行，形成完整输出图像
            samples = torch.cat(samples, dim=2)

            # 应用反缩放因子并返回结果
            return 1. / self.scale_factor * samples

        # 模式2：批量自回归(AR2)模式 - 每次处理多个块
        elif inference and ar2:  # block_batch>1
            block_size = self.image_block_size  # 图像块大小
            vit_block_size = block_size // self.patch_size  # ViT块大小
            block_h, block_w = h // block_size, w // block_size  # 块的行数和列数
            rope_position_ids = rope_position_ids.view(-1, h // self.patch_size, w // self.patch_size, 2)  # 重塑位置ID

            samples = []  # 存储每一行的处理结果
            cached = [None] * block_w  # 缓存每列(对于编程矩阵来说列，对于实际图像是行)的处理结果

            # 批量块大小计算
            block_bsize = block_size * block_batch  # 批量图像块大小
            vit_block_bsize = vit_block_size * block_batch  # 批量ViT块大小

            # 确保块数量可以被batch_size整除
            assert block_h % block_batch == 0
            assert block_w % block_batch == 0

            # 计算批量块的行数和列数
            block_batch_h = block_h // block_batch
            block_batch_w = block_w // block_batch

            # 设置迭代范围
            range_i = range(block_batch_h)
            range_j = range(block_batch_w)

            # 按行遍历批量块
            for i in range_i:
                previous = None  # 存储上一个批量块的处理结果
                sample_row = []  # 存储当前行的处理结果

                # 按列遍历批量块
                for j in range_j:
                    # 提取当前批量块的图像数据
                    tmp_images = images[:, :, i * block_bsize:(i + 1) * block_bsize,
                                 j * block_bsize:(j + 1) * block_bsize]

                    # 提取当前批量块的位置ID
                    tmp_position_ids = rope_position_ids[:, i * vit_block_bsize:(i + 1) * vit_block_bsize,
                                       j * vit_block_bsize:(j + 1) * vit_block_bsize].contiguous().view(-1,
                                                                                                        vit_block_bsize * vit_block_bsize,
                                                                                                        2)

                    # 更新kwargs中的参数
                    kwargs["images"] = tmp_images * c_in
                    kwargs["sigmas"] = c_noise.reshape(-1)
                    kwargs["rope_position_ids"] = tmp_position_ids

                    # 初始化mems列表，用于存储记忆值
                    mems = [None, None, None]

                    # 定义辅助函数，用于提取记忆键值对的前两个元素
                    def get_top_concat(x):
                        return [{"mem_kv": y['mem_kv'][:2]} for y in x]

                    # 如果不是第一列，使用前一个批量块的结果和前一列的缓存
                    if j != 0:
                        mems[0] = previous
                        if cached[j - 1] is not None:
                            mems[1] = cached[j - 1]

                    # 如果当前列有缓存，使用它
                    if cached[j] is not None:
                        mems[2] = cached[j]

                    # 执行模型前向传播
                    output, *output_per_layers = self.model_forward(*args, hw=[vit_block_bsize,
                                                                               vit_block_bsize],
                                                                    mems=mems, inference=2, **kwargs)

                    # 应用输出缩放系数和跳跃连接
                    output = output * c_out + tmp_images[:, :self.out_channels] * c_skip

                    # 处理记忆值以减少内存使用
                    if previous is not None:
                        previous = get_top_concat(previous)

                    # 更新缓存
                    if j != 0:
                        cached[j - 1] = previous

                    # 根据位置更新缓存或previous变量
                    if j == block_w - 1:  # 如果是行的最后一个批量块
                        cached[j] = get_top_concat(output_per_layers)
                    else:
                        previous = output_per_layers

                    # 将处理后的批量块加入当前行结果
                    sample_row.append(output)

                # 在水平方向上拼接当前行的所有批量块
                sample_row = torch.cat(sample_row, dim=3)
                samples.append(sample_row)

            # 在垂直方向上拼接所有行，形成完整输出图像
            samples = torch.cat(samples, dim=2)

            # 应用反缩放因子并返回结果
            return 1. / self.scale_factor * samples

        # 模式3：全图处理模式 - 一次性处理整个图像
        else:
            # 更新kwargs中的参数，用于全图处理
            kwargs["images"] = images * c_in
            kwargs["sigmas"] = c_noise.reshape(-1)

            # 设置默认处理方向为左上到右下(lt)
            direction = "lt"

            # 根据随机方向选项和训练/推理状态调整方向
            # 如果在训练阶段且随机值>0.5，方向为右下到左上(rb)
            if self.random_direction and torch.rand(1) > 0.5 and not inference:
                direction = "rb"

            # 如果在推理阶段，根据采样步骤调整方向
            # 奇数步骤使用从右下到左上的方向
            if self.random_direction and sample_step is not None and sample_step % 2 == 1:
                direction = "rb"

            # 注释掉的方向切换代码
            # if direction == "rb":
            #     direction = "lt"
            # else:
            #     direction = "rb"

            # 将方向参数添加到kwargs中
            kwargs["direction"] = direction

            # 执行模型前向传播，处理整个图像
            output, *output_per_layers = self.model_forward(*args, hw=[h // self.patch_size, w // self.patch_size],
                                                            rope_position_ids=rope_position_ids,
                                                            lr_imgs=lr_imgs, **kwargs)

            # 应用输出缩放系数和跳跃连接
            output = output * c_out + images[:, :self.out_channels] * c_skip

            # 如果启用了注意力收集，保存每层的输出用于分析
            if self.collect_attention is not None:
                self.collect_attention.append(output_per_layers)

            # 应用反缩放因子并返回结果
            # output = output.to(in_type)  # 注释掉的类型转换
            return 1. / self.scale_factor * output

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        out = self.first_stage_model.decode(z)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x):
        z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    def _init_first_stage(self):
        self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        if self.is_decoder:
            sigmas = kwargs["sigmas"]
            self.text_encoder = self.text_encoder.to(self.text_encoder.encoder_dtype)
            with torch.no_grad():
                context, context_mask = self.text_encoder(**kwargs['text_inputs'])
            if self.no_crossmask:
                context_mask = torch.ones_like(context_mask)
            kwargs['encoder_outputs'] = context.to(sigmas.dtype)
            kwargs['cross_attention_mask'] = context_mask.to(sigmas.dtype)

        device = kwargs["images"].device
        # add position_ids
        position_ids = torch.zeros(self.num_patches, 2, device=device)
        position_ids[:, 0] = torch.arange(self.num_patches) // (self.image_size // self.patch_size)
        position_ids[:, 1] = torch.arange(self.num_patches) % (self.image_size // self.patch_size)
        if self.random_position:
            # change to 15 for long context
            position_ids[:, 0] += torch.randint(0, 7 * (self.image_size // self.patch_size) + 1, (1,), device=device)
            position_ids[:, 1] += torch.randint(0, 7 * (self.image_size // self.patch_size) + 1, (1,), device=device)
        position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), kwargs["images"].shape[0], dim=0).long()
        position_ids[position_ids==-1] = 0

        return self.precond_forward(rope_position_ids=position_ids, inference=False, *args, **kwargs)

    def sample(
            self,
            shape,  # 输出张量的形状
            rope_position_ids=None,  # 旋转位置编码的位置ID
            num_steps=None,  # 采样步骤数
            images=None,  # 输入图像（可能是初始噪声或低分辨率图像）
            lr_imgs=None,  # 低分辨率输入图像
            init_noise=True,  # 是否初始化噪声
            dtype=torch.float32,  # 计算的数据类型
            device=torch.device('cuda'),  # 计算设备
            return_attention_map=False,  # 是否返回注意力图
            image_2=None,  # 可选的第二输入图像
            do_concat=True,  # 是否连接低分辨率图像
            ar=False,  # 是否使用自回归模式
            ar2=False,  # 是否使用第二种自回归模式
            block_batch=1,  # 块批处理大小
    ):
        print(f"[DEBUG] use_taylorseer = {self.use_taylorseer}")

        # 如果没有提供输入图像，则创建随机噪声作为起始点
        if images is None:
            images = torch.randn(*shape).to(dtype).to(device)

        # 初始化条件和无条件字典
        cond = {}  # 条件字典，用于指导扩散过程
        uncond = {}  # 无条件字典，用于分类器引导采样

        # 如果提供了第二张图像，将其添加到条件字典中
        if image_2 is not None:
            cond["image2"] = image_2

        # 如果有图像编码器，处理低分辨率图像并创建嵌入
        if self.image_encoder:
            if image_2 is not None:
                print("has image2!!!")
                image_embedding = self.image_encoder(image_2)
            else:
                # 对低分辨率图像进行编码，并添加文本
                image_embedding = self.image_encoder(lr_imgs, add_text=True)
            # 将图像嵌入添加到条件和无条件字典中
            cond["vector"] = image_embedding
            uncond["vector"] = image_embedding

        # 将低分辨率图像添加到条件和无条件字典中
        cond["lr_imgs"] = lr_imgs
        uncond["lr_imgs"] = lr_imgs

        # 用于分类器引导采样的连接标志（条件为1，无条件为0）
        cond["concat"] = torch.ones(1, dtype=torch.bool, device=images.device)
        uncond["concat"] = torch.zeros(1, dtype=torch.bool, device=images.device)

        # 将输入图像添加到连接条件中
        cond["concat_lr_imgs"] = images
        uncond["concat_lr_imgs"] = images

        # 计算图像补丁数量和位置编码
        h, w = images.shape[2:4]
        num_patches = h * w // (self.patch_size ** 2)
        if rope_position_ids is None:
            # 创建2D位置编码（行和列）
            position_ids = torch.zeros(num_patches, 2, device=images.device)
            position_ids[:, 0] = torch.arange(num_patches, device=images.device) // (w // self.patch_size)
            position_ids[:, 1] = torch.arange(num_patches, device=images.device) % (w // self.patch_size)

            # 对位置ID进行模运算，确保在有效范围内
            position_ids = position_ids % (8 * self.image_size // self.patch_size)

            # 复制位置ID到批次中的每个样本
            position_ids = torch.repeat_interleave(position_ids.unsqueeze(0), images.shape[0], dim=0).long()
            position_ids[position_ids == -1] = 0
            rope_position_ids = position_ids

        # 启用Transformer输出隐藏状态
        self.transformer.output_hidden_states = True

        # 定义去噪函数，使用预条件前向传播
        def denoiser(images, sigmas, rope_position_ids, cond, sample_step):
            print(f"[DEBUG] cond['concat'] = {cond['concat']}")

            step = sample_step or 0

            # ✅ 判断当前是否是 cond 分支
            # is_cond = cond.get("concat", None) is not None and cond["concat"].item()
            is_cond = cond.get("concat", None) is not None and cond["concat"].sum().item() > 0

            # ✅ 是否启用 Taylor 预测
            use_taylor = (
                    self.use_taylorseer and
                    is_cond and
                    step >= self.taylorseer.max_order and
                    self.taylorseer.should_use_taylor(step)
            )

            if use_taylor:
                print(f"[Taylor] step {step}: using Taylor prediction")
                eps = self.taylorseer.predict(
                    images, t=step, sigmas=sigmas,
                    rope_position_ids=rope_position_ids, cond=cond
                )
            else:
                eps = self.precond_forward(
                    images=images,
                    sigmas=sigmas,
                    rope_position_ids=rope_position_ids,
                    inference=True,
                    sample_step=step,
                    do_concat=do_concat,
                    ar=ar,
                    ar2=ar2,
                    block_batch=block_batch,
                    **cond
                )

                # ✅ 只缓存 cond 分支的 eps
                if self.use_taylorseer and is_cond:
                    if len(self.taylorseer.cache_t) == 0 or self.taylorseer.cache_t[0] != step:
                        self.taylorseer.cache_eps.appendleft(eps.detach())
                        self.taylorseer.cache_t.appendleft(step)
                        print(f"[DEBUG] cached step {step}")

            # 🔍 调试日志
            print(f"[DEBUG] step = {step}, is_cond = {is_cond}, use_taylor = {use_taylor}")
            if self.use_taylorseer:
                print(f"[DEBUG] cache_t = {list(self.taylorseer.cache_t)}")

            return eps

        # 如果需要返回注意力图，初始化注意力收集列表
        if return_attention_map:
            self.collect_attention = []

        # 执行采样过程
        samples = self.sampler(
            denoiser=denoiser,
            x=None,
            cond=cond,
            uc=uncond,
            num_steps=num_steps,
            rope_position_ids=rope_position_ids,
            init_noise=init_noise
        )

        # 如果有第一阶段模型，解码生成的样本
        if self.first_stage_model:
            samples = self.decode_first_stage(samples)

        # 禁用Transformer输出隐藏状态
        self.transformer.output_hidden_states = False

        # 返回结果，可能包括注意力图
        if return_attention_map:
            attention_maps = self.collect_attention
            self.collect_attention = None
            return samples, attention_maps

        if self.use_taylorseer and self.taylorseer.test_flops:
            self.taylorseer.report_flops()

        return samples
