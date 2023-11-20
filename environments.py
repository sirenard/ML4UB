import ecole
import pyscipopt

import utils


class SimpleBranchingDynamics(ecole.dynamics.BranchingDynamics):
    def reset_dynamics(self, model):
        # keeping a reference to the model
        # Fix huge problems
        self.pyscipopt_model: pyscipopt.Model = model.as_pyscipopt()

        utils.configure(self.pyscipopt_model)

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)

class HistoryBranchingEnvironment(ecole.environment.Environment):
    __Dynamics__ = SimpleBranchingDynamics
    __DefaultObservationFunction__ = ecole.observation.NodeBipartite



class SimpleConfiguringDynamics(ecole.dynamics.ConfiguringDynamics):
    def reset_dynamics(self, model):
        # Share memory with Ecole model
        self.pyscipopt_model: pyscipopt.Model = model.as_pyscipopt()

        utils.configure(self.pyscipopt_model)

        # Let the parent class get the model to the root node and return
        # the done flag / action_set
        return super().reset_dynamics(model)


class SimpleBranching(ecole.environment.Environment):
    __Dynamics__ = SimpleBranchingDynamics

class SimpleConfiguring(ecole.environment.Configuring):
    __Dynamics__ = SimpleConfiguringDynamics