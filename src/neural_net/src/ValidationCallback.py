from tensorflow.keras.callbacks import Callback

class ValidationCallback(Callback):
    
    def __init__(self, xy_validation):
        self.xy_validation = xy_validation
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.evaluate(self.xy_validation)
    
    
