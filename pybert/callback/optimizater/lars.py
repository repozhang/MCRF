import torch
from torch.optim.optimizer import Optimizer

class Lars(Optimizer):

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, scale_clip=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Lars, self).__init__(params, defaults)
        # LARS arguments
        self.scale_clip = scale_clip
        if self.scale_clip is None:
            self.scale_clip = (0, 10)

    def __setstate__(self, state):
        super(Lars, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # LARS
                p_norm = p.data.pow(2).sum().sqrt()
                update_norm = d_p.pow(2).sum().sqrt()
                # Compute the local LR
                if p_norm == 0 or update_norm == 0:
                    local_lr = 1
                else:
                    local_lr = p_norm / update_norm

                p.data.add_(-group['lr'] * local_lr, d_p)

        return loss