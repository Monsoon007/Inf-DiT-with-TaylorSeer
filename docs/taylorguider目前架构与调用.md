

Search files...
现在我需要查找与调用 TaylorSeerGuider 相关的代码，特别是 denoise 方法：

Search files...
现在我需要看一下 DiffusionEngine 中调用 sampler 的相关代码：

Search files...
让我再查看一下和 prepare_inputs 和调用相关的代码：

Search files...
最后，让我们查看一下 model.py 中 Line 1123 的上下文：

Read file: dit/model.py
基于收集到的信息，我将整理出一份关于 TaylorSeerGuider 及其相关调用代码的详细文档：

# TaylorSeerGuider 及相关调用代码文档

## 1. TaylorSeerGuider 核心类实现

```python
# 文件位置: dit/sampling/guiders.py
class TaylorSeerGuider:
    def __init__(self, scale=5.0, use_cache=True, step_threshold=0.05):
        self.scale = scale
        self.use_cache = use_cache
        self.step_threshold = step_threshold
        self.cache = {}        # step_key -> ε
        self.history = []      # [(sigma, ε)]
        self._skip_cur_step = False  # 临时标记当前是否跳过模型推理
        self._identity_mode = (scale == 0.0)  # 是否等价于 IdentityGuider

        if self._identity_mode:
            print("[TaylorSeerGuider] ⚠️  scale=0 → Running in IdentityGuider mode (single batch, no guidance)")

        # 加入统计项
        self.total_steps = 0
        self.skipped_steps = 0

    def has_cache_for(self, sigma):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma
        return step_key in self.cache

    def prepare_inputs(self, x, sigma, cond, uc, rope_position_ids):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma

        self.total_steps += 1  # 每次 prepare_inputs 都算一次 step

        if self.use_cache and self.has_cache_for(sigma):
            self._skip_cur_step = True
            return None, None, None, None
        else:
            self._skip_cur_step = False

        if self._identity_mode:
            # 与 IdentityGuider 一致：只使用 cond，不拼接
            c_out = {k: cond[k] for k in cond}
            return x, sigma, c_out, rope_position_ids

        # 标准拼接流程：cond + uncond
        c_out = {k: torch.cat((uc[k], cond[k]), dim=0) for k in cond}
        x_cat = torch.cat([x] * 2, dim=0)
        sigma_cat = torch.cat([sigma] * 2, dim=0)
        if rope_position_ids is not None:
            rope_position_ids = torch.cat([rope_position_ids] * 2, dim=0)

        return x_cat, sigma_cat, c_out, rope_position_ids

    def __call__(self, x, sigma):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma

        # 优先使用 cache
        if self.use_cache and step_key in self.cache:
            print(f"[TaylorSeer] ⚡ Using cached ε at sigma={step_key:.4f}")
            return self.cache[step_key]

        if x is None:
            raise ValueError("x is None but no cache hit! This shouldn't happen.")

        if self._identity_mode:
            print(f"[TaylorSeer] ✅ IdentityGuider mode at sigma={step_key:.4f}")
            return x  # 与 IdentityGuider 一致

        # 正常引导：CFG 合成
        x_u, x_c = x.chunk(2)
        guided = x_u + self.scale * (x_c - x_u)
        print(f"[TaylorSeer] ✅ Real model output at sigma={step_key:.4f}")

        if self.use_cache:
            self.history.append((step_key, guided.detach()))
            self.cache[step_key] = guided.detach()

            if len(self.history) >= 2:
                s0, e0 = self.history[-2]
                s1, e1 = self.history[-1]
                delta = abs(s1 - s0)
                if delta < self.step_threshold:
                    s2 = s1 - delta
                    extrapolated = e1 + (e1 - e0)
                    self.cache[s2] = extrapolated.detach()
                    print(f"[TaylorSeer] ⏩ Precomputed extrapolation for sigma={s2:.4f}")

        return guided

    def get_stats(self):
        skip_ratio = self.skipped_steps / max(self.total_steps, 1)
        return {
            "total_steps": self.total_steps,
            "skipped_steps": self.skipped_steps,
            "skip_ratio": skip_ratio,
        }
```

## 2. denoise 方法实现（BaseDiffusionSampler）

```python
# 文件位置: dit/sampling/samplers.py
def denoise(self, x, denoiser, sigma, cond, uc, rope_position_ids, sample_step=None):
    # 准备输入（可能提前判断 skip）
    images, sigmas, cond, rope_position_ids = self.guider.prepare_inputs(x, sigma, cond, uc, rope_position_ids)

    # 如果为 IdentityGuider 则没有启用成功 taylorguider
    if self.guider.__class__.__name__ == "IdentityGuider":
        print("没有启用成功taylorguider")

    if getattr(self.guider, "_skip_cur_step", False): 
        # ⚡ 跳过模型推理
        print(self.guider._skip_cur_step)
        denoised = self.guider(None, sigma)
    else:
        # 🧠 正常调用 denoiser（DiT 模型）
        denoised = denoiser(images, sigmas, rope_position_ids, cond, sample_step) # 会调用 dit/model.py:1118
        denoised = self.guider(denoised, sigma)

    return denoised
```

## 3. wrapped_denoiser 函数 (DiffusionEngine.sample 方法内)

```python
# 文件位置: dit/model.py
def wrapped_denoiser(images, sigmas, rope_position_ids, cond, sample_step):
    if hasattr(self, 'guider') and self.guider is not None:
        print(f"[DiffusionEngine.sample] Using guider: {self.guider}")
        # guider 预处理输入
        x, sigma, c, rope_ids = self.guider.prepare_inputs(images, sigmas, cond, None, rope_position_ids)
        if x is None:
            return None

        # 构造 kwargs：全部通过字典传参，避免重复
        c = c.copy()
        c['images'] = x
        c['sigmas'] = sigma

        # 调用去噪接口（不再显式传递 lr_imgs 等）
        denoised = self.precond_forward(
            inference=0,
            rope_position_ids=rope_ids,
            ar=ar,
            ar2=ar2,
            sample_step=sample_step,
            block_batch=block_batch,
            **c
        )
        return self.guider(denoised, sigma)

    else:
        print(f"[DiffusionEngine.sample] No guider available, using default denoising")
        # 默认路径，同样只通过 kwargs 传参
        cond = cond.copy()
        cond['images'] = images
        cond['sigmas'] = sigmas

        return self.precond_forward(
            inference=0,
            rope_position_ids=rope_position_ids,
            ar=ar,
            ar2=ar2,
            sample_step=sample_step,
            block_batch=block_batch,
            **cond
        )
```

## 4. 采样主循环 (DiffusionEngine.sample 方法)

```python
# 文件位置: dit/model.py
def sample(self, shape, rope_position_ids=None, num_steps=None, images=None, lr_imgs=None,
           init_noise=True, dtype=torch.float32, device=torch.device('cuda'),
           return_attention_map=False, image_2=None, do_concat=True, ar=False,
           ar2=False, block_batch=1):
    # ✅ 显示 guider 信息
    print(f"[DiffusionEngine.sample] Starting sampling with guider: {getattr(self, 'guider', None)}")
    
    # ... [初始化输入部分省略]
    
    # 启用Transformer输出隐藏状态
    self.transformer.output_hidden_states = True
    
    # 定义 wrapped_denoiser 函数
    # ... [前文已列出]
    
    # 执行采样过程
    samples = self.sampler(
        denoiser=wrapped_denoiser,
        x=None,
        cond=cond,
        uc=uncond,
        num_steps=num_steps,
        rope_position_ids=rope_position_ids,
        init_noise=init_noise
    )
    
    # ... [后处理部分省略]
    
    return samples
```

## 5. TaylorSeerGuider 初始化与配置

```python
# 文件位置: dit/model.py 中的 _build_modeling 方法
def _build_modeling(self, args, modeling_configs):
    # ... [其他代码省略]
    
    sampler_config = modeling_configs.pop('sampler_config')
    
    # Debug logging for sampler config
    print(f"[DiffusionEngine] Initial sampler config: {sampler_config}")
    
    # 确保 sampler_config 有 params 字段
    if 'params' not in sampler_config:
        sampler_config['params'] = {}
    
    # 确保 params 中有 guider_config
    if 'guider_config' not in sampler_config['params']:
        sampler_config['params']['guider_config'] = {}
    
    # 设置 guider
    if getattr(args, "guider", None):
        guider_target = f"dit.sampling.guiders.{args.guider}"
        sampler_config['params']['guider_config']['target'] = guider_target
        print(f"[DiffusionEngine] ✅ Setting guider in sampler config: {guider_target}")
        print(f"[DiffusionEngine] Updated sampler config: {sampler_config}")
        
    if getattr(args, "guiderscale", None) is not None:
        sampler_config['params']['guider_config']['params'] = sampler_config['params']['guider_config'].get('params', {})
        sampler_config['params']['guider_config']['params']['scale'] = args.guiderscale
        print(f"[DiffusionEngine] ✅ Setting guider scale in sampler config: {args.guiderscale}")
        print(f"[DiffusionEngine] Updated sampler config: {sampler_config}")
    
    # 实例化 sampler
    self.sampler = instantiate_from_config(sampler_config)
    print(f"[DiffusionEngine] Sampler initialized: {self.sampler}")
```

## 6. 命令行参数配置

```python
# 文件位置: generate_t2i_sr.py
def add_sample_specific_args(parser):
    # ... [其他参数省略]
    
    # ✅ 添加 guider 参数
    group.add_argument('--guider', type=str, default=None,
                       help='Override guider class (e.g. VanillaCFG, IdentityGuider, TaylorSeerGuider)')
    group.add_argument('--guiderscale', type=float, default=None, help='Override CFG/TaylorSeer scale')
    
    return parser
```

## 7. 跳步判断流程

1. **入口点**：当 `BaseDiffusionSampler.denoise()` 被调用时，首先调用 `guider.prepare_inputs`
   
2. **判断是否跳步**：
   ```python
   if self.use_cache and self.has_cache_for(sigma):
       self._skip_cur_step = True
       return None, None, None, None
   ```
   
3. **跳步执行**：
   ```python
   if getattr(self.guider, "_skip_cur_step", False):
       denoised = self.guider(None, sigma)
   ```
   
4. **缓存命中**：在 `__call__` 方法中，如果传入 `x=None`，会从缓存中获取结果：
   ```python
   if self.use_cache and step_key in self.cache:
       return self.cache[step_key]
   ```

5. **缓存生成**：每次执行真实计算后，将结果保存到缓存并可能通过外推生成额外缓存：
   ```python
   if len(self.history) >= 2:
       # ... 计算外推预测
       extrapolated = e1 + (e1 - e0)
       self.cache[s2] = extrapolated.detach()
   ```

## 8. 完整调用流程

1. 通过命令行参数 `--guider TaylorSeerGuider --guiderscale 值` 启用 TaylorSeer

2. DiffusionEngine 初始化时，将这些参数传递给 sampler 配置

3. 采样时的执行顺序：
   - `DiffusionEngine.sample()`
   - → 创建 `wrapped_denoiser` 函数
   - → 调用 `self.sampler(denoiser=wrapped_denoiser, ...)`
   - → 采样器内部循环调用 `denoise(x, denoiser, sigma, ...)`
   - → 调用 `guider.prepare_inputs` 判断是否可以跳步
   - → 如果可以跳步，直接返回缓存结果；否则执行真实计算

4. 跳步条件：
   - 当前 sigma 值在缓存中存在
   - 使用启用了缓存 (use_cache=True)

5. 缓存生成方式：
   - 真实计算后保存到缓存
   - 当连续两个时间步的 sigma 差值小于阈值时，外推计算下一个时间步
