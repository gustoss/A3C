import numpy as np
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box


class PreprocessImage(ObservationWrapper):
    
    def __init__(self, env, height = 64, width = 64, grayscale = True):
        super(PreprocessImage, self).__init__(env)
        self.img_size = (width, height)
        self.grayscale = grayscale
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [width, height, n_colors])

    def _observation(self, img):
        img = imresize(img, self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims = True)
        img = img.astype('float32') / 255.
        return img
