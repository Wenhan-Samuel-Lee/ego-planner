import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
from PIL import Image

import math


# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
# complete this to fit into nn
class ImageSequence(Sequence):

    def __init__(self, x_set, batch_size, img_shape, hash_table, offset=0):
        self.x = x_set
        self.batch_size = batch_size
        self.img_shape = img_shape
        print('batch is {}'.format(self.batch_size))
        self.hash_table = hash_table
        self.offset = offset
        
    def __len__(self):
        num_batches = len(self.x) / self.batch_size
        
        return int(num_batches)
    
    def __getitem__(self, idx): 
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        img_x = np.zeros((self.batch_size, *self.img_shape, 1))

        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
#             image = Image.open('../dataset_transformed_12_7/img_' + str(self.hash_table[i] + self.offset) + '.jpg'
#             imagearr = np.asarray(image).reshape(*self.img_shape, 1) 
            imagearr = np.zeros((480, 640, 1))
            
            img_x[i % self.batch_size] = imagearr / 255

        return img_x, img_x
        
class UAVSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size, img_shape, hash_table, offset=0):
        self.x, self.y = x_set, y_set
        
        self.batch_size = batch_size
        self.img_shape = img_shape
        print('batch is {}'.format(self.batch_size))
        self.hash_table = hash_table
        self.offset = offset
        
    def __len__(self):
        num_batches = len(self.x) / self.batch_size
        
        return int(num_batches)
    
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        img_x = np.zeros((self.batch_size, *self.img_shape, 1))

        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
#             image = Image.open('../dataset_transformed_12_7/img_' + str(self.hash_table[i] + self.offset) + '.jpg')
#             imagearr = np.asarray(image).reshape(*self.img_shape, 1) 
            
            imagearr = np.zeros((480, 640, 1))
            img_x[i % self.batch_size] = imagearr / 255

        return [img_x, np.array(batch_x)], np.array(batch_y)
        