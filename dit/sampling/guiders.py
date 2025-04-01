import os
from functools import partial
import math
import json

import torch

from dit.utils import default, instantiate_from_config

from PIL import Image
class VanillaCFG:
    """
    implements parallelized CFG
    """

    def __init__(self, scale, dyn_thresh_config=None):
        scale_schedule = lambda scale, sigma: scale  # independent of step
        self.scale_schedule = partial(scale_schedule, scale)
        self.dyn_thresh = instantiate_from_config(
            default(
                dyn_thresh_config,
                {
                    "target": "dit.sampling.utils.NoDynamicThresholding"
                },
            )
        )


    def __call__(self, x, sigma):
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        x_pred = self.dyn_thresh(x_u, x_c, scale_value)
        return x_pred

    def prepare_inputs(self, x, s, c, uc, rope_position_ids):
        c_out = dict()

        for k in c:
            c_out[k] = torch.cat((uc[k], c[k]), 0)
        
        if rope_position_ids is not None:
            rope_position_ids = torch.cat([rope_position_ids] * 2)
        
        return torch.cat([x] * 2), torch.cat([s] * 2), c_out, rope_position_ids


class IdentityGuider:
    def __call__(self, x, sigma):
        return x

    def prepare_inputs(self, x, s, c, uc, rope_position_ids):
        c_out = dict()

        for k in c:
            c_out[k] = c[k]

        return x, s, c_out, rope_position_ids


class TaylorSeerGuider:
    def __init__(self, scale=5.0, use_cache=True, step_threshold=0.05, interval=4, max_order=2):
        self.scale = scale
        print(f"[TaylorSeerGuider] 初始化引导器: scale={scale}, use_cache={use_cache}, interval={interval}, max_order={max_order}")
        
        self.use_cache = use_cache
        self.interval = interval
        self.max_order = max_order
        self.total_steps = 0
        self.skipped_steps = 0
        self.num_steps = None  # 初始化 num_steps
        self._identity_mode = (scale == 0.0)  # 是否等价于 IdentityGuider
        self._skip_cur_step = False  # 临时标记当前是否跳过模型推理
        
        if self._identity_mode:
            print("[TaylorSeerGuider] ⚠️  scale=0 → Running in IdentityGuider mode (single batch, no guidance)")
            
        self.cache = {
            'noise_cache': {},
            'history': [],
            'activated_steps': [],
            'cache_counter': 0
        }
        print("[TaylorSeerGuider] 缓存系统已初始化")

    def has_cache_for(self, sigma):
        """检查是否有对应 sigma 值的缓存"""
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma
        return step_key in self.cache['noise_cache']

    def prepare_inputs(self, x, sigma, cond, uc, rope_position_ids):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma

        self.total_steps += 1  # 每次 prepare_inputs 都算一次 step

        if self.use_cache and self.has_cache_for(sigma):
            self._skip_cur_step = True
            self.skipped_steps += 1
            print(f"[TaylorSeerGuider] 使用缓存跳过步骤 {step_key:.4f}")
            return None, None, None, None
        else:
            self._skip_cur_step = False
            # 更新缓存计数器
            self.cache['cache_counter'] = (self.cache['cache_counter'] + 1) % self.interval
            if self._should_activate(step_key):
                self.cache['activated_steps'].append(step_key)
                print(f"[TaylorSeerGuider] 激活步骤 {step_key:.4f} (计数器: {self.cache['cache_counter']})")

        if self._identity_mode:
            # 与 IdentityGuider 一致：只使用 cond，不拼接
            c_out = {k: cond[k] for k in cond}
            return x, sigma, c_out, rope_position_ids

        # 标准拼接流程：cond + uncond
        if uc is None:
            raise ValueError("When scale > 0, uc cannot be None")
            
        c_out = {k: torch.cat((uc[k], cond[k]), dim=0) for k in cond}
        x_cat = torch.cat([x] * 2, dim=0)
        sigma_cat = torch.cat([sigma] * 2, dim=0)
        if rope_position_ids is not None:
            rope_position_ids = torch.cat([rope_position_ids] * 2, dim=0)

        return x_cat, sigma_cat, c_out, rope_position_ids

    def _should_activate(self, step_key):
        if self.num_steps is None:
            return True
        
        # 判断是否需要全激活的条件
        last_steps = (step_key <= 2)  # 最后几步全激活
        first_steps = (step_key >= (self.num_steps - 2))  # 修正：使用 >= 而不是 >
        fresh_interval = self.interval
        
        should_activate = last_steps or first_steps or (self.cache['cache_counter'] == fresh_interval - 1)
        if should_activate:
            print(f"[TaylorSeerGuider] 步骤 {step_key:.4f} 需要全激活: last_steps={last_steps}, first_steps={first_steps}, interval={self.cache['cache_counter'] == fresh_interval - 1}")
        return should_activate

    def __call__(self, x, sigma):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma
        
        # 如果当前步骤被跳过，直接返回缓存的结果
        if self._skip_cur_step and step_key in self.cache['noise_cache']:
            print(f"[TaylorSeerGuider] 从缓存返回步骤 {step_key:.4f} 的结果")
            return self.cache['noise_cache'][step_key]
        
        # 更新缓存
        if step_key in self.cache['activated_steps']:
            with torch.no_grad():
                self.cache['noise_cache'][step_key] = x.detach()
                self.cache['history'].append(x.detach())
                if len(self.cache['history']) > self.max_order:
                    self.cache['history'].pop(0)
                print(f"[TaylorSeerGuider] 更新缓存: 步骤 {step_key:.4f}, 历史大小: {len(self.cache['history'])}")
        
        # 如果引导尺度为0，直接返回输入
        if self.scale == 0:
            return x
            
        # 否则应用引导
        x_u, x_c = x.chunk(2)
        scale_value = self.scale_schedule(sigma)
        return x_c + scale_value * (x_c - x_u)

    def get_stats(self):
        skip_ratio = self.skipped_steps / max(self.total_steps, 1)
        stats = {
            "mode": "TaylorSeerGuider",
            "scale": self.scale,
            "total_steps": self.total_steps,
            "skipped_steps": self.skipped_steps,
            "skip_ratio": f"{skip_ratio:.2%}",
            "cache_size": len(self.cache['noise_cache']),
            "history_size": len(self.cache['history']),
            "activated_steps": len(self.cache['activated_steps']),
            "last_activated_step": self.cache['activated_steps'][-1] if self.cache['activated_steps'] else None,
            "cache_counter": self.cache['cache_counter'],
            "interval": self.interval
        }
        print(f"[TaylorSeerGuider] 统计信息: {json.dumps(stats, indent=2)}")
        return stats

    def _predict_with_taylor(self, x, sigma, cond, uc, rope_position_ids):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma
        
        if len(self.cache['activated_steps']) < 2:
            print("[TaylorSeerGuider] 警告：历史数据不足，无法进行泰勒预测")
            return x, sigma, cond, rope_position_ids
        
        difference_distance = self.cache['activated_steps'][-1] - self.cache['activated_steps'][-2]
        if difference_distance == 0:
            print("[TaylorSeerGuider] 警告：时间步差为0，跳过泰勒预测")
            return x, sigma, cond, rope_position_ids
        
        # 构建泰勒因子
        taylor_factors = {}
        taylor_factors[0] = self.cache['noise_cache'][self.cache['activated_steps'][-1]]
        
        # 计算高阶差分
        for i in range(self.max_order):
            if len(self.cache['history']) >= i + 1:
                taylor_factors[i + 1] = (taylor_factors[i] - self.cache['history'][-i-1]) / difference_distance
            else:
                break
        
        # 应用泰勒公式
        x_diff = step_key - self.cache['activated_steps'][-1]
        predicted = 0
        for i in range(len(taylor_factors)):
            predicted += (1 / math.factorial(i)) * taylor_factors[i] * (x_diff ** i)
        
        c_out = {k: torch.cat((uc[k], cond[k]), dim=0) for k in cond}
        x_cat = torch.cat([predicted] * 2, dim=0)
        sigma_cat = torch.cat([sigma] * 2, dim=0)
        if rope_position_ids is not None:
            rope_position_ids = torch.cat([rope_position_ids] * 2, dim=0)
        
        return x_cat, sigma_cat, c_out, rope_position_ids

    def set_num_steps(self, num_steps):
        """设置总步数"""
        self.num_steps = num_steps
        print(f"[TaylorSeerGuider] 设置总步数: {num_steps}")

    def clear_cache(self):
        """清理缓存，在开始新的采样时调用"""
        self.cache = {
            'noise_cache': {},
            'history': [],
            'activated_steps': [],
            'cache_counter': 0
        }
        self.total_steps = 0
        self.skipped_steps = 0
        self._skip_cur_step = False
