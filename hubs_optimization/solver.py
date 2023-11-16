from typing import Union, TypeAlias
from dimod import DiscreteQuadraticModel
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridDQMSampler
from dwave.system import LeapHybridCQMSampler

QuadraticModel: TypeAlias = Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]


class QuadraticModelAnnealingSolver:
    def __init__(self, discrete: bool = False):
        self.discrete = discrete
        self.solver = self._initiate_solver()

    def _initiate_solver(self):
        if self.discrete:
            return LeapHybridDQMSampler()
        else:
            return LeapHybridCQMSampler()

    def sample(self, quadratic_model: QuadraticModel):
        if self.discrete:
            return self.solver.sample_dqm(
                quadratic_model, label='DQM Airline Hubs')
        else:
            return self.solver.sample_cqm(
                quadratic_model, label='CQM Airline Hubs')
