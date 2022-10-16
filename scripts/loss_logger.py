
class LossLogger:
    def __init__(self, **kwargs):
        self._initial_losses = kwargs
        self.reset_losses()

    def reset_losses(self):
        self._count = 0
        self._running_losses = self._initial_losses.copy()

    def add_losses(self, **kwargs):
        self._count += 1
        for key, value in kwargs.items():
            self._running_losses[key] += value.item()

    def dispaly_losses(self, logger, step):
        tensorboard = logger.experiment
        for key, value in self._running_losses.items():
            tensorboard.add_scalar(key, value / self._count, step)
        self.reset_losses()

