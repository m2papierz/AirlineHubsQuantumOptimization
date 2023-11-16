from typing import Union, TypeAlias
from abc import abstractmethod
from dimod import DiscreteQuadraticModel
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridDQMSampler
from dwave.system import LeapHybridCQMSampler


QuadraticModel: TypeAlias = Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]


class QuadraticModelSolver:
    def __init__(self):
        self.solver = self._initiate_solver()

    @abstractmethod
    def _initiate_solver(self):
        pass

    @abstractmethod
    def sample(self, quadratic_model: QuadraticModel):
        pass


class AnnealingSolver(QuadraticModelSolver):
    def __init__(self, discrete: bool = False):
        self.discrete = discrete
        self.label = 'CQM Airline Hubs'
        super().__init__()

    def _initiate_solver(self):
        if self.discrete:
            return LeapHybridDQMSampler()
        else:
            return LeapHybridCQMSampler()

    def sample(self, quadratic_model: QuadraticModel):
        if self.discrete:
            return self.solver.sample_dqm(
                quadratic_model, label=self.label)
        else:
            return self.solver.sample_cqm(
                quadratic_model, label=self.label)
