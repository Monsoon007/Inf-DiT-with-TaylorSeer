import os
from functools import partial

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

        # ✅ 加入统计项
        self.total_steps = 0
        self.skipped_steps = 0

    def has_cache_for(self, sigma):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma
        return step_key in self.cache

    def prepare_inputs(self, x, sigma, cond, uc, rope_position_ids):
        step_key = sigma[0].item() if isinstance(sigma, torch.Tensor) else sigma

        self.total_steps += 1  # ✅ 每次 prepare_inputs 都算一次 step

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
