import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Tuple


HUB_COLOR = '#109010'
REGULAR_COLOR = '#afafaf'


def get_node_data(
        graph: nx.Graph,
        hub_nodes: List[str],
        regular_nodes_size: int = 100
) -> Tuple[list, list, dict]:
    degrees = dict(graph.degree)

    node_color = [
        HUB_COLOR if node in hub_nodes else REGULAR_COLOR
        for node in graph.nodes()
    ]

    node_size = [
        regular_nodes_size * degrees[node] * 2 if node in hub_nodes
        else regular_nodes_size * degrees[node]
        for node in graph.nodes()
    ]

    node_labels = {
        node: node if (node in hub_nodes) else ''
        for node in graph.nodes()
    }

    return node_color, node_size, node_labels


def get_edge_data(
        graph: nx.Graph,
        hub_edges: List[int],
        regular_edges_width: int = 1
) -> Tuple[list, list]:

    edge_color = [
        HUB_COLOR if edge in hub_edges else REGULAR_COLOR
        for edge in graph.edges()
    ]

    edge_width = [
        regular_edges_width * 2 if edge in hub_edges else regular_edges_width
        for edge in graph.edges()
    ]

    return edge_color, edge_width


def draw_solution_graph(
        nodes: Dict[str, List[float]],
        edges: List[Tuple[str, str]],
        hub_nodes: List[str],
        hub_edges: List[int],
        save_dir: Path
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(nodes.keys())
    graph.add_edges_from(edges)

    # Get nodes data
    node_color, node_size, node_labels = get_node_data(
        graph=graph, hub_nodes=hub_nodes
    )

    # Get edges data
    edge_color, edge_width = get_edge_data(
        graph=graph, hub_edges=hub_edges
    )

    # Draw graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G=graph, pos=nodes, node_color=node_color,
        node_size=node_size, labels=node_labels,
        edge_color=edge_color, width=edge_width
    )

    plt.savefig(save_dir / 'best_solution_graph')
