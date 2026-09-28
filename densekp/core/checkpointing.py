"""Save ranked checkpoints only after validation of the current weights."""
from pytorch_lightning.callbacks import ModelCheckpoint


class ValidationCheckpoint(ModelCheckpoint):
    """Save at the first validation at least ``every_n_steps`` after the last save.

    Steps mean Lightning global_step (optimizer attempts). AMP can skip a real
    parameter update while this counter still advances.

    A train-batch callback sees the previous validation's metric. Saving there
    would associate a stale score with new weights. Epoch-end saves are disabled.
    """
    def __init__(self, every_n_steps, **kwargs):
        if every_n_steps < 1:
            raise ValueError('GENERAL.CHECKPOINT_STEPS must be positive')
        self.every_n_steps = every_n_steps
        self.last_validation_save_step = 0
        super().__init__(every_n_train_steps=0, every_n_epochs=1,
                         save_on_train_epoch_end=False, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step == 0:
            return
        if trainer.global_step - self.last_validation_save_step < self.every_n_steps:
            return
        self.last_validation_save_step = trainer.global_step
        super().on_validation_end(trainer, pl_module)

    def state_dict(self):
        state = super().state_dict()
        state['last_validation_save_step'] = self.last_validation_save_step
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.last_validation_save_step = state_dict.get('last_validation_save_step', 0)
