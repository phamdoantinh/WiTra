"""Project the realized AdamW floor-head displacement relative to a floor-only reference.

Encoder, other heads, and UW follow the normal MAIN update. AdamW moments are
committed from the MAIN gradient; the reference never commits optimizer state.
This protects a local first-order floor-loss comparison, not accuracy or XY loss.
"""
from __future__ import annotations

import torch


class FloorUpdateProtection:
    def __init__(self, floor_head, optimizer, max_norm):
        if type(optimizer) is not torch.optim.AdamW:
            raise ValueError('Floor update protection requires torch.optim.AdamW')
        self.parameters = list(floor_head.parameters())
        if not self.parameters:
            raise ValueError('Empty floor head')
        group = next(g for g in optimizer.param_groups
                     if any(p is self.parameters[0] for p in g['params']))
        if not all(any(p is q for q in group['params']) for p in self.parameters):
            raise ValueError('Floor head must belong to one AdamW group')
        if any(group.get(k, False) for k in ('fused', 'capturable', 'differentiable', 'maximize')):
            raise ValueError('Unsupported AdamW execution option for floor protection')
        self.optimizer, self.group, self.max_norm = optimizer, group, max_norm
        self.model_parameters = list(group['params'])
        self.shadow = [torch.nn.Parameter(p.detach().clone()) for p in self.parameters]
        options = {k: group[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach')}
        self.reference = torch.optim.AdamW(self.shadow, **options)

    def floor_gradient(self, floor_loss):
        # autograd.grad leaves .grad empty and retains the same forward/dropout.
        result = torch.autograd.grad(floor_loss, self.parameters, retain_graph=True)
        if not all(torch.isfinite(g).all() for g in result):
            raise FloatingPointError('Non-finite floor probe gradient')
        return tuple(g.detach() for g in result)

    @torch.no_grad()
    def prepare_reference(self, floor_gradient, floor_coefficient):
        # Call after MAIN backward, BEFORE global clipping or optimizer.step.
        if any(p.grad is None for p in self.parameters):
            raise ValueError('Missing floor-head MAIN gradients')
        weighted = [g * floor_coefficient for g in floor_gradient]
        # Norm of a full reference gradient: replace only the floor-head block.
        head_ids = {id(p) for p in self.parameters}
        norm_terms = [p.grad.double().square().sum() for p in self.model_parameters
                      if id(p) not in head_ids and p.grad is not None]
        norm_terms += [g.double().square().sum() for g in weighted]
        reference_norm = torch.stack(norm_terms).sum().sqrt()
        # Match torch clip_grad_norm_'s coefficient convention. UW not clipped.
        coefficient = (self.max_norm / (reference_norm + 1e-6)).clamp(max=1)
        self.before = [p.detach().clone() for p in self.parameters]
        self.u = torch.cat([g.reshape(-1).double() for g in floor_gradient])
        for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach'):
            self.reference.param_groups[0][key] = self.group[key]
        self.reference.state.clear()
        for source, target, old, gradient in zip(self.parameters, self.shadow, self.before, weighted):
            target.copy_(old)
            target.grad = (gradient * coefficient).clone()
            source_state = self.optimizer.state.get(source, {})
            self.reference.state[target] = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in source_state.items()}
        self.reference.step()
        self.reference_displacement = torch.cat([
            (old - new.detach()).reshape(-1).double()
            for old, new in zip(self.before, self.shadow)])

    @torch.no_grad()
    def apply(self):
        # Call once after the real MAIN optimizer.step. Moments stay MAIN's.
        main = torch.cat([(old - p).reshape(-1).double()
                          for old, p in zip(self.before, self.parameters)])
        norm2 = self.u.square().sum()
        gap = torch.dot(self.u, self.reference_displacement - main)
        # Tiny norms are skipped and reported; no claim of an exact projection there.
        valid = norm2 > 1e-24
        amount = torch.where(valid, gap.clamp(min=0) / norm2.clamp(min=1e-24),
                             torch.zeros_like(gap))
        correction = amount * self.u
        offset = 0
        for parameter in self.parameters:
            n = parameter.numel()
            parameter.sub_(correction[offset:offset+n].reshape_as(parameter).to(parameter.dtype))
            offset += n
        realized = torch.cat([(old - p).reshape(-1).double()
                              for old, p in zip(self.before, self.parameters)])
        margin = torch.dot(self.u, realized - self.reference_displacement)
        # Finite precision can leave a very small negative margin. Report it.
        values = torch.stack([torch.ones_like(gap), ((gap > 0) & valid).double(),
                              (~valid).double(), gap.clamp(min=0), margin,
                              (realized-main).norm()])
        if not torch.isfinite(values).all():
            raise FloatingPointError('Non-finite projected floor update')
        return values
