import numpy as np

from typing import Dict, List
from dimod import DiscreteQuadraticModel

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
            airports_data: Dict[str, List[float]],
            max_hubs: int,
            hub_discount: float,
            first_constraint_lagrange: float,
            second_constraint_lagrange: float
    ):
        self._data = airports_data
        self._max_hubs = max_hubs
        self._discount = hub_discount
        self._airports_num = len(airports_data)

        self._first_lm = first_constraint_lagrange
        self._second_lm = second_constraint_lagrange

        self._demand_m = self._create_demand_matrix()
        self._cost_m = self._calculate_cost_matrix()

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

    def _not_hub_connected_to_hub_constraint(
            self,
            dqm: DiscreteQuadraticModel
    ) -> None:
        for origin in range(self._airports_num):
            for destination in range(self._airports_num):
                if origin != destination:
                    dqm.set_linear_case(
                        v=origin, case=destination,
                        bias=dqm.get_linear_case(
                            v=origin, case=destination
                        ) + self._first_lm
                    )
                    dqm.set_quadratic_case(
                        u=origin, u_case=destination,
                        v=destination, v_case=destination,
                        bias=dqm.get_quadratic_case(
                            u=origin, u_case=destination,
                            v=destination, v_case=destination
                        ) - self._first_lm
                    )

    def _max_hubs_constraint(
            self,
            dqm: DiscreteQuadraticModel
    ) -> None:
        for origin in range(self._airports_num):
            dqm.set_linear_case(
                v=origin, case=origin,
                bias=dqm.get_linear_case(
                    v=origin, case=origin
                ) + (1 - 2 * self._max_hubs) * self._second_lm
            )

            for destination in range(self._airports_num):
                if origin != destination:
                    dqm.set_quadratic_case(
                        u=origin, u_case=origin,
                        v=destination, v_case=destination,
                        bias=dqm.get_quadratic_case(
                            u=origin, u_case=origin,
                            v=destination, v_case=destination
                        ) + self._second_lm
                    )

    def _add_objective_function(
            self,
            dqm: DiscreteQuadraticModel
    ) -> None:
        for origin in range(self._airports_num):
            for destin in range(self._airports_num):
                for first_hub in range(self._airports_num):
                    operating_cost = self._cost_m[origin][first_hub] * self._demand_m[origin][destin]
                    dqm.set_linear_case(
                        v=origin, case=first_hub,
                        bias=dqm.get_linear_case(
                            v=origin, case=first_hub
                        ) + operating_cost
                    )

                    operating_cost = self._cost_m[destin][first_hub] * self._demand_m[origin][destin]
                    dqm.set_linear_case(
                        v=destin, case=first_hub,
                        bias=dqm.get_linear_case(
                            v=destin, case=first_hub
                        ) + operating_cost
                    )

                    for second_hub in range(self._airports_num):
                        if origin != destin:
                            operating_cost = (1 - self._discount) * self._cost_m[
                                first_hub][second_hub] * self._demand_m[origin][destin]
                            dqm.set_quadratic_case(
                                u=origin, u_case=first_hub,
                                v=destin, v_case=second_hub,
                                bias=dqm.get_quadratic_case(
                                    u=origin, u_case=first_hub,
                                    v=destin, v_case=second_hub
                                ) + operating_cost
                            )

    def build_discrete_quadratic_model(self):
        dqm = DiscreteQuadraticModel()

        for airport in range(self._airports_num):
            dqm.add_variable(
                num_cases=self._airports_num, label=airport
            )

        # Constraint 1: Every node that is not a hub must be connected to a hub.
        self._not_hub_connected_to_hub_constraint(dqm)

        # Constraint 2: There must be exactly 'max_hubs' hubs.
        self._max_hubs_constraint(dqm)

        # Set the objective function - minimizing the overall operating cost.
        self._add_objective_function(dqm)

        return dqm

    def get_solution_cost(self, sample):
        sample_cost = 0
        num_airports = self._demand_m.shape[0]

        for origin in range(num_airports):
            for destin in range(origin + 1, num_airports):
                first_hub = sample[origin]
                second_hub = sample[destin]

                sample_cost += self._demand_m[origin][destin] * (
                        self._cost_m[origin][first_hub] +
                        self._cost_m[destin][second_hub] +
                        (1 - self._discount) * self._cost_m[first_hub][second_hub]
                )

        return sample_cost
