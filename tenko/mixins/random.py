"""
Mixin functionality for random numbers.
"""

from toolbox.numpy import random


class RandomMixin(object):

    def set_random_seed(self, seed):
        """
        Set random seed on `rnd` RandomState attribute based on a string seed.
        """
        klass = self.__class__.__name__
        seed = klass if seed is None else seed
        self.seed = sum(list(map(ord, seed)))
        self.rnd = random.RandomState(seed=self.seed)

        if hasattr(self, 'out'):
            self.out(f'Instance seed: {self.seed} [key: \'{seed}\']')

    def set_default_random_seed(self, seed):
        """
        Set the default numpy random seed according to a key string or hash.
        """
        klass = self.__class__.__name__
        seed = klass if seed is None else seed
        self.seed = sum(list(map(ord, seed)))
        random.seed(self.seed)
        self.rnd = random.random_sample

        if hasattr(self, 'out'):
            self.out(f'Default seed: {self.seed} [key: \'{seed}\']')
