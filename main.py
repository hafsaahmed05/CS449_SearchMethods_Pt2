"""
main.py  — Entry point for AI Search Visualizer
"""

import argparse
from visualization import Visualizer
from graph_loader import load_city_graph
from grid import GridEnvironment


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coord_file", default="coordinates.csv")
    p.add_argument("--adj_file",   default="Adjacencies.txt")
    return p.parse_args()


def main():
    args = get_args()

    _city_cache = {}
    def load_city():
        if "obj" not in _city_cache:
            _city_cache["obj"] = load_city_graph(args.coord_file, args.adj_file)
        return _city_cache["obj"]

    def make_grid(rows, cols, obs, conn):
        obs = max(0.20, min(0.30, obs))
        return GridEnvironment(rows=rows, cols=cols,
                               obstacle_pct=obs, connectivity=conn)

    Visualizer(
        graph_loader_fn=load_city,
        grid_factory_fn=make_grid,
    )


if __name__ == "__main__":
    main()