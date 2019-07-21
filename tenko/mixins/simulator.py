"""
Mixin functionality for simulator contexts.
"""


class SimulatorMixin(object):

    def set_simulation_parameters(self, **params):
        """
        Set simulation parameters in global scope and shared state.
        """
        Global = self._get_global_scope()
        State = self._get_state()

        Global.update(params)
        State.update(params)

        self.write_json(params, 'simulation')

        self.out('Simulation parameters:')
        for name, value in params.items():
            self.out(f'- {name} = {value}', hideprefix=True)

        # Put the context itself into shared state for project-wide access
        State.update(context=self)
