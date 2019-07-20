"""
Mixin functionality for simulator contexts.
"""

from toolbox.numpy import random


class SimulatorMixin(object):

    def set_simulation_parameters(self, **params):
        """
        Set simulation parameters in global scope and shared state.
        """
        self._get_global_scope().update(params)
        self._get_state().update(params)
        self.write_json(params, 'simulation')

        self.out('Simulation parameters:')
        for name, value in params.items():
            self.out(f'- {name} = {value}', hideprefix=True)

        # Write JSON file with the simulation parameters

    def set_random_seed(self, key):
        """
        Set the numpy random seed according to a key string or hash.
        """
        newseed = sum(list(map(ord, key)))
        random.seed(newseed)
        self.out(f'{newseed} [key: \'{key}\']', prefix='RandomSeed')
