from fastai.vision import *
from fastai.callbacks import *

__all__ = ["SaveModelCallback"]

class SaveModelCallback(TrackerCallback):
    "SaveModelCallback modified for distributed transfer learning - remove torch.load"
    def __init__(self, learn:Learner, monitor:str='val_loss', mode:str='auto', every:str='improvement', name:str='bestmodel', best_init=None):
        super().__init__(learn, monitor=monitor, mode=mode)
        self.every,self.name = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        if best_init: self.best = best_init 
      
    def on_train_begin(self, **kwargs:Any)->None:
        "Initializes the best value."
        if not hasattr(self, 'best'):
            self.best = float('inf') if self.operator == np.less else -float('inf')
#         print('best init score:', self.best) 
        
    def jump_to_epoch(self, epoch:int)->None:
        try: 
            self.learn.load(f'{self.name}_{epoch-1}', purge=False)
            print(f"Loaded {self.name}_{epoch-1}")
        except: print(f'Model {self.name}_{epoch-1} not found.')

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.learn.save(f'{self.name}_{epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.learn.save(f'{self.name}')