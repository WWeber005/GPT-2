from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]
                # si c'est la première fois que on lance, il faut donc initier:
                if len(state) == 0:
                    state["step"] = 0
                    state["m_t"] =  torch.zeros_like(p.data)
                    state["v_t"] = torch.zeros_like(p.data)
                    
                    
                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]
                # beta hyperparameters from the `group` dictionary.
                betas = group["betas"]
                beta_1 = betas[0]
                beta_2 = betas[1]
                
                ### Apply bias correction
                ###(using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                state["step"]+= 1
                t = state["step"]
                m_t = state["m_t"]
                v_t = state["v_t"]

                m_t = beta_1 * m_t + (1-beta_1)*grad
                v_t = beta_2 * v_t + (1-beta_2)*grad**2
                state["m_t"] = m_t
                state["v_t"] = v_t
                if group["correct_bias"]:
                    m_hat = m_t / (1-beta_1**t)
                    v_hat = v_t / (1-beta_2**t)
                else:
                    m_hat = m_t
                    v_hat = v_t
                p.data = p.data - alpha*m_hat/(torch.sqrt(v_hat)+group["eps"])
                p.data = p.data - alpha*group["weight_decay"]*p.data


        return loss
