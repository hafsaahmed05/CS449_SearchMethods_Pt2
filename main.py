"""
main.py  — Entry point for AI Search Visualizer
Boots the unified single-window UI.
All settings (mode, algo, heuristic, start/goal, speed) are controlled
inside the window — no CLI needed for normal use.

Optional flags:
  --coord_file  path to coordinates CSV   (default: coordinates.csv)
  --adj_file    path to adjacency txt     (default: Adjacencies.txt)
  --benchmark   run batch benchmark and exit (no GUI)
"""

import argparse
from visualization import Visualizer
from graph_loader import load_city_graph
from grid import GridEnvironment


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coord_file", default="coordinates.csv")
    p.add_argument("--adj_file",   default="Adjacencies.txt")
    p.add_argument("--benchmark",  action="store_true")
    return p.parse_args()


def main():
    args = get_args()

    if args.benchmark:
        from benchmark import batch_compare
        def env_factory(seed):
            return GridEnvironment(rows=10, cols=10,
                                   obstacle_pct=0.25, connectivity=4, seed=seed)
        batch_compare(env_factory)
        return

    # ── City graph loader (lazy — only called when user picks city mode) ──
    _city_cache = {}
    def load_city():
        if "obj" not in _city_cache:
            _city_cache["obj"] = load_city_graph(args.coord_file, args.adj_file)
        return _city_cache["obj"]

    # ── Grid factory ──────────────────────────────────────────────────────
    def make_grid(rows, cols, obs, conn):
        obs = max(0.20, min(0.30, obs))
        return GridEnvironment(rows=rows, cols=cols,
                               obstacle_pct=obs, connectivity=conn)

    # ── Launch unified window (blocks until closed) ───────────────────────
    Visualizer(
        graph_loader_fn=load_city,
        grid_factory_fn=make_grid,
    )


if __name__ == "__main__":
    main()