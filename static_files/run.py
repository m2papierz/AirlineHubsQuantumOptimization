import sys
import argparse

from pathlib import Path
from typing import Dict, List
from hubs_optimization.airline_hubs_problem import AirlineHubsProblem
from utils import PathsManager

sys.path.insert(1, str(Path(__file__).parent.parent))


def load_airports_data(
        data_path: Path
) -> Dict[str, List[float]]:
    airport_info = {}

    with open(data_path, 'r') as file:
        for line in file.readlines():
            data = line.split(",")
            airport_name = data[1]
            airport_latitude = float(data[2])
            airport_longitude = float(data[3].strip())

            airport_info[airport_name] = [
                -airport_longitude, airport_latitude
            ]

    return airport_info


def interpret_results(sample, airport_names):
    hubs = []
    node_to_hub_connections = []

    for origin, destination in sample.items():
        if origin == destination:
            hubs.append(airport_names[origin])
        else:
            node_to_hub_connections.append((airport_names[origin], airport_names[destination]))

    return hubs, node_to_hub_connections


def check_result_validity(hubs, node_to_hub_connections, max_hubs):
    if len(hubs) != max_hubs:
        return False

    i = 0
    while i < len(node_to_hub_connections):
        origin, destination = node_to_hub_connections[i]

        if destination not in hubs:
            return False
        i += 1

    return True


def main():
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser(description="aaa")

    parser.add_argument(
        '--solver', type=str, default=config.solver, choices=['aqc', 'qaoa', 'gas'],
        help='')
    parser.add_argument(
        '--aqc_type', type=str, default=config.aqc_type, choices=['qpu', 'hybrid', 'exact'],
        help='')
    parser.add_argument(
        '--max_hubs', type=int, default=config.max_hubs,
        help='')
    parser.add_argument(
        '--hub_discount', type=float, default=config.hub_discount,
        help='')
    parser.add_argument(
        '--first_lagrange', type=float, default=config.first_lagrange,
        help='')
    parser.add_argument(
        '--second_lagrange', type=float, default=config.second_lagrange,
        help='')
    args = parser.parse_args()

    # Create problem instance and its discrete quadratic model
    airports_data = load_airports_data(
        data_path=paths_manager.data_dir() / 'airports.txt')

    problem = AirlineHubsProblem(
        airports_data=airports_data,
        max_hubs=args.max_hubs,
        hub_discount=args.hub_discount,
        first_constraint_lagrange=args.first_lagrange,
        second_constraint_lagrange=args.second_lagrange
    )
    dqm = problem.build_discrete_quadratic_model()


if __name__ == "__main__":
    main()
