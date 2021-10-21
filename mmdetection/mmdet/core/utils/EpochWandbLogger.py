from mmcv.runner.hooks.logger import WandbLoggerHook
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class EpochWandbLogger(WandbLoggerHook):
    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True):
        super().__init__(init_kwargs, interval, ignore_last, reset_flag, commit, by_epoch, with_step)
        
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            tags['epoch'] = self.get_epoch(runner)
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=(self.commit and self.get_mode(runner) == 'train'))
            else:
                self.wandb.log(tags, commit=self.commit and self.get_mode(runner) == 'train')