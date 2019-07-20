"""
Mixin functionality for simulator contexts.
"""


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
