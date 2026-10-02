"""Parameter snapshots and pre-update gradient checks for bounded training audits."""
import pytorch_lightning as pl
import torch


def probes(model):
    """Clone up to 4096 scalars from the first backbone/head parameter to CPU.

    Keys and parameter traversal order are historical audit contracts. Values
    are float32, detached snapshots; this function preserves model mode and RNG.
    It is a bounded update probe, not proof that every parameter has changed.
    """
    return {name: next(module.parameters()).detach().flatten()[:4096].float().cpu().clone()
            for name, module in (('backbone', model.backbone), ('head', model.head))}


class CheckUpdates(pl.Callback):
    """Record finite-gradient update attempts before the optimizer runs.

    Missing all gradients in either module, or a nonfinite present gradient,
    aborts the update. Individual parameters may have no gradient, as before.
    ``attempts`` records the pre-step global_step and gradient tensor counts;
    callers must separately check completed optimizer updates. This callback
    retains Lightning's default empty state_dict: attempts are per run, not
    restored checkpoint history.
    """
    def __init__(self):
        self.attempts = []

    def on_before_optimizer_step(self, trainer, model, optimizer):
        groups = {}
        for name, module in (('backbone', model.backbone), ('head', model.head)):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError(f'Missing/nonfinite {name} gradients; stop before update')
            groups[name] = len(gradients)
        self.attempts.append(dict(step=trainer.global_step, finite_gradient_tensors=groups))
        print('Finite optimizer update ' + str(trainer.global_step), flush=True)
