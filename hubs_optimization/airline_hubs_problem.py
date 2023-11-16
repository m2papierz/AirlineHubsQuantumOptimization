import numpy as np

from abc import abstractmethod, ABC
from typing import Dict, List, Union, Tuple
from dimod import DiscreteQuadraticModel
from dimod import BinaryQuadraticModel
from dimod import ConstrainedQuadraticModel

EARTH_RADIUS_IN_KM = 6371
np.random.seed(42)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    """

    """
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    d_lat, d_lon = lat2 - lat1, lon2 - lon1

    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c * EARTH_RADIUS_IN_KM


class AirlineHubsProblem:
    def __init__(
            self,
            airports_data: Tuple[Dict[str, List[float]], np.ndarray, np.ndarray],
            max_hubs: int,
            hub_discount: float,
            first_constraint_lagrange: float,
            second_constraint_lagrange: float,
            custom_data: bool = False
    ):
        self._data = airports_data[0]
        self._max_hubs = max_hubs
        self._discount = hub_discount
        self._airports_num = len(airports_data[0])

        self._first_lm = first_constraint_lagrange
        self._second_lm = second_constraint_lagrange

        if not custom_data:
            self._cost_m = airports_data[1]
            self._demand_m = airports_data[2]
        else:
            self._cost_m = self._calculate_cost_matrix()
            self._demand_m = self._create_demand_matrix()

    @property
    def max_hubs(self):
        return self._max_hubs

    def _calculate_cost_matrix(self):
        cost_matrix = np.zeros((self._airports_num, self._airports_num))
        airport_names = list(self._data.keys())

        for i in range(self._airports_num):
            for j in range(i + 1, self._airports_num):
                lat1, lon1 = self._data[airport_names[i]]
                lat2, lon2 = self._data[airport_names[j]]
                cost = haversine(lat1, lon1, lat2, lon2)
                cost_matrix[i][j] = cost
                cost_matrix[j][i] = cost

        return cost_matrix

    def _create_demand_matrix(self):
        upper_triangular = np.triu(np.random.rand(
            self._airports_num, self._airports_num))
        symmetric_matrix = upper_triangular + upper_triangular.T
        np.fill_diagonal(symmetric_matrix, 0)
        normalized_matrix = symmetric_matrix / symmetric_matrix.sum()
        return normalized_matrix

    @abstractmethod
    def _not_hub_connected_to_hub_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        """
        The first constraint ensures that each non-hub node is connected to a
        hub node, which can be identified by its connections to other hubs.
        """
        pass

    @abstractmethod
    def _max_hubs_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        """
        The second constraint ensures that there are no more hubs than 'max_hubs'.
        """
        pass

    @abstractmethod
    def _add_objective_function(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        """
        Defines the central objective function of the problem.

        It is important to note that demand is always related to the demand between
        the origin and destination of each traveler, regardless of the intermediate
        airports.
        """
        pass

    @abstractmethod
    def build_quadratic_model(self):
        pass

    def get_solution_cost(self, sample):
        sample_cost = 0
        num_airports = self._demand_m.shape[0]

        for o in range(num_airports):
            for d in range(o + 1, num_airports):
                first_hub = sample[o]
                second_hub = sample[d]

                sample_cost += self._demand_m[o][d] * (
                        self._cost_m[o][first_hub] +
                        self._cost_m[d][second_hub] +
                        (1 - self._discount) * self._cost_m[first_hub][second_hub]
                )

        return sample_cost


class AirlineHubsProblemDiscrete(AirlineHubsProblem, ABC):
    def __init__(
            self,
            airports_data: Tuple[Dict[str, List[float]], np.ndarray, np.ndarray],
            max_hubs: int,
            hub_discount: float,
            first_constraint_lagrange: float,
            second_constraint_lagrange: float
    ):
        super().__init__(
            airports_data=airports_data,
            max_hubs=max_hubs,
            hub_discount=hub_discount,
            first_constraint_lagrange=first_constraint_lagrange,
            second_constraint_lagrange=second_constraint_lagrange
        )

    def _not_hub_connected_to_hub_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        for o in range(self._airports_num):
            for d in range(self._airports_num):
                if o != d:
                    existing_coefficient = quadratic_model.get_linear_case(v=o, case=d)
                    quadratic_model.set_linear_case(
                        v=o, case=d,
                        bias=existing_coefficient + self._first_lm
                    )

                    existing_coefficient = quadratic_model.get_quadratic_case(
                        u=o, u_case=d, v=d, v_case=d)
                    quadratic_model.set_quadratic_case(
                        u=o, u_case=d, v=d, v_case=d,
                        bias=existing_coefficient - self._first_lm
                    )

    def _max_hubs_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        for o in range(self._airports_num):
            existing_coefficient = quadratic_model.get_linear_case(v=o, case=o)
            quadratic_model.set_linear_case(
                v=o, case=o,
                bias=existing_coefficient + (1 - 2 * self._max_hubs) * self._second_lm
            )

            for d in range(self._airports_num):
                if o != d:
                    existing_coefficient = quadratic_model.get_quadratic_case(
                        u=o, u_case=o, v=d, v_case=d)
                    quadratic_model.set_quadratic_case(
                        u=o, u_case=o, v=d, v_case=d,
                        bias=existing_coefficient + self._second_lm
                    )

    def _add_objective_function(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        for o in range(self._airports_num):
            for d in range(self._airports_num):
                for first_hub in range(self._airports_num):

                    # Total cost of the flights going from the origin to its hub, calculated as
                    # the total demand for that connection, times the cost of each unit of flow.
                    existing_coefficient = quadratic_model.get_linear_case(v=o, case=first_hub)
                    operating_cost = self._cost_m[o][first_hub] * self._demand_m[o][d]
                    quadratic_model.set_linear_case(
                        v=o, case=first_hub,
                        bias=existing_coefficient + operating_cost
                    )

                    # Total cost of the flights going to a final destination from its hub,
                    # calculated as the total demand for that connection, times the cost of each
                    # unit of flow.
                    existing_coefficient = quadratic_model.get_linear_case(v=d, case=first_hub)
                    operating_cost = self._cost_m[d][first_hub] * self._demand_m[o][d]
                    quadratic_model.set_linear_case(
                        v=d, case=first_hub,
                        bias=existing_coefficient + operating_cost
                    )

                    # Total cost of the flights between hubs, calculated as the total demand for
                    # that connection, times the cost of each unit of flow for this part of the
                    # journey, times one minus the discount applied to flights between hubs.
                    for second_hub in range(self._airports_num):
                        if o != d:
                            existing_coefficient = quadratic_model.get_quadratic_case(
                                u=o, u_case=first_hub, v=d, v_case=second_hub)
                            operating_cost = (1 - self._discount) * self._cost_m[
                                first_hub][second_hub] * self._demand_m[o][d]
                            quadratic_model.set_quadratic_case(
                                u=o, u_case=first_hub, v=d, v_case=second_hub,
                                bias=existing_coefficient + operating_cost
                            )

    def build_quadratic_model(self):
        quadratic_model = DiscreteQuadraticModel()

        for airport in range(self._airports_num):
            # Define each x_i for every airport 'i', with two possible cases:
            #   1. If x_i = j and i != j, then 'i' is not a hub
            #   2. If x_i = i, then 'i' is a hub
            quadratic_model.add_variable(
                num_cases=self._airports_num, label=airport
            )

        # First constraint: Every node that is not a hub must be connected to a hub
        self._not_hub_connected_to_hub_constraint(quadratic_model)

        # Second constraint: There must be exactly 'max_hubs' hubs
        self._max_hubs_constraint(quadratic_model)

        # Set the objective function for minimizing the overall operating cost
        self._add_objective_function(quadratic_model)

        return quadratic_model


class AirlineHubsProblemBinary(AirlineHubsProblem, ABC):
    def __init__(
            self,
            airports_data: Tuple[Dict[str, List[float]], np.ndarray, np.ndarray],
            max_hubs: int,
            hub_discount: float,
            first_constraint_lagrange: float,
            second_constraint_lagrange: float
    ):
        super().__init__(
            airports_data=airports_data,
            max_hubs=max_hubs,
            hub_discount=hub_discount,
            first_constraint_lagrange=first_constraint_lagrange,
            second_constraint_lagrange=second_constraint_lagrange
        )

    def _not_hub_connected_to_hub_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        for o in range(self._airports_num):
            for d in range(self._airports_num):
                if o != d:
                    constraint = BinaryQuadraticModel('BINARY')
                    constraint.add_linear(
                        v=(o, d), bias=self._first_lm)
                    constraint.add_quadratic(
                        u=(o, d), v=(d, d), bias=-self._first_lm)
                    quadratic_model.add_constraint(constraint == 0)

    def _max_hubs_constraint(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        linear_terms = {(i, i): 1.0 for i in range(self._airports_num)}
        constraint = BinaryQuadraticModel('BINARY')
        constraint.add_linear_from(linear_terms)
        quadratic_model.add_constraint(constraint == self._max_hubs, label='num hubs')

    def _add_objective_function(
            self,
            quadratic_model: Union[DiscreteQuadraticModel, ConstrainedQuadraticModel]
    ) -> None:
        M = np.sum(self._demand_m, axis=0) + np.sum(self._demand_m, axis=1)
        Q = (1 - self._discount) * np.kron(self._demand_m, self._cost_m)

        linear = (M * self._cost_m.T).T.flatten()

        objective = BinaryQuadraticModel(linear, Q, 'BINARY')
        objective.relabel_variables({
            idx: (i, j) for idx, (i, j) in
            enumerate((i, j) for i in range(self._airports_num) for j in range(self._airports_num))
        })

        quadratic_model.set_objective(objective)

    def build_quadratic_model(self):
        quadratic_model = ConstrainedQuadraticModel()

        # Set the objective function for minimizing the overall operating cost
        self._add_objective_function(quadratic_model)

        # Add constraint to make variables discrete
        for v in range(self._airports_num):
            quadratic_model.add_discrete([(v, i) for i in range(self._airports_num)])

        # First constraint: Every node that is not a hub must be connected to a hub
        self._not_hub_connected_to_hub_constraint(quadratic_model)

        # Second constraint: There must be exactly 'max_hubs' hubs
        self._max_hubs_constraint(quadratic_model)

        return quadratic_model
