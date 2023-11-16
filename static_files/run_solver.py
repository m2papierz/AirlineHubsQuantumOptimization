import sys
import argparse
import itertools
import numpy as np

from pathlib import Path
from typing import Dict, List, Tuple, Union, TypeAlias

from hubs_optimization.solver import QuadraticModelAnnealingSolver
from hubs_optimization.airline_hubs_problem import AirlineHubsProblemBinary
from hubs_optimization.airline_hubs_problem import AirlineHubsProblemDiscrete
from hubs_optimization.graph_plotting import draw_solution_graph
from utils import PathsManager, load_file_data

sys.path.insert(1, str(Path(__file__).parent.parent))

AirlineHubsProblem: TypeAlias = Union[AirlineHubsProblemBinary, AirlineHubsProblemDiscrete]


def load_airports_data(
        data_dir: Path
) -> Tuple[Dict[str, List[float]], np.ndarray, np.ndarray]:
    cost = load_file_data(file_path=data_dir / 'cost.csv')
    demand = load_file_data(file_path=data_dir / 'demand.csv')
    demand = demand / np.sum(np.sum(demand))

    airport_info = {}
    with open(data_dir / 'airports.txt', 'r') as file:
        for line in file.readlines():
            data = line.split(",")
            airport_name = data[1]
            airport_latitude = float(data[2])
            airport_longitude = float(data[3].strip())

            airport_info[airport_name] = [
                -airport_longitude, airport_latitude
            ]

    return airport_info, cost, demand


def interpret_results(
        sample_set,
        airport_names: List[str],
        discrete: bool
) -> Tuple[Dict, List[str], List[Tuple[str, str]]]:
    n = len(airport_names)
    hubs = []
    hubs_connections = []

    if not discrete:
        sample_set = sample_set.filter(lambda d: d.is_feasible).aggregate()
        assignments = [
            {i: j for i in range(n) for j in range(n) if sample.sample[i, j] == 1}
            for sample in sample_set.data()
        ]
        best_sample = assignments[0]
    else:
        best_sample = sample_set.first.sample

    for key, val in best_sample.items():
        if key == val:
            hubs.append(airport_names[key])
        else:
            hubs_connections.append((airport_names[key], airport_names[val]))

    return best_sample, hubs, hubs_connections


def find_best_solution(
        data: Dict[str, List[float]],
        problem: AirlineHubsProblem,
        solver: QuadraticModelAnnealingSolver,
        reports_dir: Path
):
    # Create discrete quadratic model and sample
    quadratic_model = problem.build_quadratic_model()
    sample_set = solver.sample(quadratic_model=quadratic_model)

    # Interpret results
    best_solution, hubs, hubs_connections = interpret_results(
        sample_set=sample_set,
        airport_names=list(data.keys()),
        discrete=isinstance(problem, AirlineHubsProblemDiscrete)
    )

    # Save the best solution data
    with open(reports_dir / 'best_solution.txt', 'w') as f:
        f.write(
            f'Hubs: {hubs}\n'
            f'Solution cost: {problem.get_solution_cost(best_solution)}\n'
            f'Solution energy: {next(sample_set.data()).energy}'
        )

    # Plot and save the best solution graph
    hub_to_hub_connections = list(itertools.combinations(hubs, 2))
    draw_solution_graph(
        nodes=data,
        edges=hub_to_hub_connections + hubs_connections,
        hub_nodes=hubs,
        hub_edges=hub_to_hub_connections,
        save_dir=reports_dir
    )


def main():
    paths_manager = PathsManager()
    config = paths_manager.config
    parser = argparse.ArgumentParser(description="aaa")

    parser.add_argument(
        '--problem', type=str, default=config.problem, choices=['DQM', 'CQM'],
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
    data = load_airports_data(
        data_dir=paths_manager.data_dir())

    if args.problem == 'CQM':
        problem = AirlineHubsProblemBinary(
            airports_data=data,
            max_hubs=args.max_hubs,
            hub_discount=args.hub_discount,
            first_constraint_lagrange=args.first_lagrange,
            second_constraint_lagrange=args.second_lagrange
        )
        solver = QuadraticModelAnnealingSolver()
    else:
        problem = AirlineHubsProblemDiscrete(
            airports_data=data,
            max_hubs=args.max_hubs,
            hub_discount=args.hub_discount,
            first_constraint_lagrange=args.first_lagrange,
            second_constraint_lagrange=args.second_lagrange
        )
        solver = QuadraticModelAnnealingSolver(discrete=True)

    find_best_solution(
        data=data[0],
        problem=problem,
        solver=solver,
        reports_dir=paths_manager.reports_dir()
    )


if __name__ == "__main__":
    main()
