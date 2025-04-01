import torch
import math
from collections import deque

class TaylorSeerWrapper:
    def __init__(self, model, interval=4, max_order=2, test_flops=False):
        self.model = model
        self.interval = interval
        self.max_order = max_order
        self.test_flops = test_flops

        self.cache_eps = deque(maxlen=max_order + 1)
        self.cache_t = deque(maxlen=max_order + 1)

        # FLOPs counters
        self.total_steps = 0
        self.taylor_steps = 0
        self.model_calls = 0

    def should_use_taylor(self, t):
        return len(self.cache_eps) > self.max_order and t % self.interval == 0

    def derivative_approx(self):
        T = list(self.cache_t)
        E = list(self.cache_eps)
        dt = [T[i] - T[i+1] for i in range(len(T)-1)]
        d1 = [(E[i] - E[i+1]) / dt[i] for i in range(len(dt))]
        if self.max_order == 1:
            return [E[0], d1[0]]
        d2 = [(d1[i] - d1[i+1]) / (dt[i] + dt[i+1]) for i in range(len(d1)-1)]
        return [E[0], d1[0], d2[0]]

    def taylor_expand(self, delta_t):
        derivs = self.derivative_approx()
        eps_pred = derivs[0]
        for i in range(1, len(derivs)):
            eps_pred = eps_pred + (delta_t ** i) / math.factorial(i) * derivs[i]
        return eps_pred

    def predict(self, x, t, sigmas=None, rope_position_ids=None, cond=None):
        self.total_steps += 1
        t_tensor = torch.tensor([t], device=x.device, dtype=torch.float32)

        if self.should_use_taylor(t):
            self.taylor_steps += 1
            if self.test_flops:
                print(f"[Taylor] step {t}: using Taylor prediction")
            delta_t = self.cache_t[0] - t
            return self.taylor_expand(delta_t)
        else:
            eps = self.model.model_forward(x, sigmas=t_tensor, rope_position_ids=rope_position_ids, **(cond or {}))
            self.cache_eps.appendleft(eps.detach())
            self.cache_t.appendleft(t)
            self.model_calls += 1
            if self.test_flops:
                print(f"[Model]  step {t}: model executed")
            return eps

    def report_flops(self):
        print("="*30)
        print("TaylorSeer FLOPs Report")
        print(f"Total Steps       : {self.total_steps}")
        print(f"Model Calls       : {self.model_calls}")
        print(f"Taylor Predictions: {self.taylor_steps}")
        print(f"Ratio Skipped     : {self.taylor_steps / self.total_steps:.2%}")
        print("="*30)
