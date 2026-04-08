"""
benchmark.py — Algorithm profiler and batch comparison tool

Claude Prompt:
Build a benchmarking harness with two modes: a single-run mode that runs one
algorithm on one environment and logs metrics, and a batch compare mode that
runs all algorithms across multiple random seeds (at least 5) and outputs a
comparison table with mean ± std for time, memory, and nodes expanded.
"""

import time
import tracemalloc
import random
import statistics

from search import bfs, dfs, iddfs, greedy_best_first, astar, reconstruct_path
from heuristics import get_heuristic

# ── Algorithm wrapper ────────────────────────────────────────

ALGORITHMS = {
    "bfs": lambda env, h_fn=None: bfs(env.start, env.goal, env.adjacency),
    "dfs": lambda env, h_fn=None: dfs(env.start, env.goal, env.adjacency),
    "iddfs": lambda env, h_fn=None: iddfs(env.start, env.goal, env.adjacency),
    "greedy": lambda env, h_fn: greedy_best_first(env.start, env.goal, env.adjacency, h_fn),
    "astar": lambda env, h_fn: astar(env.start, env.goal, env.adjacency, h_fn),
}


# ── Core runner: one algorithm on one env ────────────────────────────────────

def run_once(env, algo_name, h_name='euclidean'):
    algo_fn = ALGORITHMS.get(algo_name)
    h_fn    = get_heuristic(h_name)

    tracemalloc.start()
    t_start = time.perf_counter()

    generator = algo_fn(env, h_fn=h_fn)

    nodes_expanded  = 0
    nodes_generated = 0
    peak_footprint  = 0
    final_state     = None
    children_counts = []

    for state in generator:
        nodes_expanded += 1

        frontier_size = len(state.get('frontier', []))
        visited_size  = len(state.get('visited', []))

        footprint = frontier_size + visited_size
        peak_footprint = max(peak_footprint, footprint)

        neighbors = env.adjacency.get(state['current'], [])
        children_counts.append(len(neighbors))
        nodes_generated += len(neighbors)

        final_state = state

        if state.get('found'):
            break

    t_end = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_ms     = (t_end - t_start) * 1000
    peak_mem_kb = peak / 1024

    # ── Path reconstruction ────────────────────────────────────────────────
    path = []
    path_cost = float('inf')
    solution_depth = 0

    if final_state and final_state.get('found'):
        path = reconstruct_path(final_state['parent'], env.start, env.goal)
        solution_depth = len(path) - 1

        cost = 0.0
        for i in range(len(path) - 1):
            if hasattr(env, 'edge_weight'):
                cost += env.edge_weight(path[i], path[i+1])
            else:
                dr = abs(path[i][0] - path[i+1][0])
                dc = abs(path[i][1] - path[i+1][1])
                cost += 1.414 if (dr + dc == 2) else 1.0

        path_cost = round(cost, 4)

    avg_b = round(statistics.mean(children_counts), 2) if children_counts else 0
    max_b = max(children_counts) if children_counts else 0

    return {
        'algo'           : algo_name,
        'h_name'         : h_name,
        'found'          : bool(final_state and final_state.get('found')),
        'time_ms'        : round(time_ms, 4),
        'peak_mem_kb'    : round(peak_mem_kb, 2),
        'algo_footprint' : peak_footprint,
        'nodes_expanded' : nodes_expanded,
        'nodes_generated': nodes_generated,
        'path_cost'      : path_cost,
        'solution_depth' : solution_depth,
        'path'           : path,
        'branching_avg'  : avg_b,
        'branching_max'  : max_b,
    }


# ── Single-run report ────────────────────────────────────────────────────────

def single_run_report(env, algo_name, h_name='euclidean'):
    result = run_once(env, algo_name, h_name)
    _print_single(result)
    return result


def _print_single(r):
    print(f"\n{'='*52}")
    print(f"  Algorithm : {r['algo'].upper()}   Heuristic: {r['h_name']}")
    print(f"{'='*52}")
    print(f"  Found              : {'YES' if r['found'] else 'NO'}")
    print(f"  Wall-clock time    : {r['time_ms']:.4f} ms")
    print(f"  Process memory peak: {r['peak_mem_kb']:.2f} KB")
    print(f"  Algo footprint peak: {r['algo_footprint']} nodes")
    print(f"  Nodes expanded     : {r['nodes_expanded']}")
    print(f"  Nodes generated    : {r['nodes_generated']}")
    print(f"  Solution depth     : {r['solution_depth']}")
    print(f"  Path cost          : {r['path_cost']}")
    print(f"  Branching (avg/max): {r['branching_avg']} / {r['branching_max']}")

    if r['path']:
        short = str(r['path'][:4])[:-1] + ', ...]' if len(r['path']) > 4 else str(r['path'])
        print(f"  Path               : {short}")

    print(f"{'='*52}")


# ── Batch comparison ─────────────────────────────────────────────────────────

def batch_compare(env_factory, algo_names=None, h_name='euclidean',
                  n_runs=5, seeds=None, label='batch'):

    if algo_names is None:
        algo_names = list(ALGORITHMS.keys())

    if seeds is None:
        seeds = [random.randint(1, 9999) for _ in range(n_runs)]

    seeds = seeds[:n_runs]

    print(f"\n{'='*60}")
    print(f"  BATCH BENCHMARK — {label.upper()} ({n_runs} runs)")
    print(f"  Seeds: {seeds}")
    print(f"{'='*60}")

    summaries = []

    for algo_name in algo_names:
        times, mems, expanded, costs = [], [], [], []

        for seed in seeds:
            env = env_factory(seed)

            if hasattr(env, "is_connected") and not env.is_connected():
                continue

            r = run_once(env, algo_name, h_name)

            if r['found']:
                times.append(r['time_ms'])
                mems.append(r['peak_mem_kb'])
                expanded.append(r['nodes_expanded'])
                costs.append(r['path_cost'])

        if not times:
            summaries.append({'algo': algo_name, 'note': 'no solved runs'})
            continue

        summaries.append({
            'algo': algo_name,
            'n_solved': len(times),
            'time_mean': round(statistics.mean(times), 4),
            'time_std': round(statistics.stdev(times) if len(times) > 1 else 0, 4),
            'mem_mean_kb': round(statistics.mean(mems), 2),
            'mem_std_kb': round(statistics.stdev(mems) if len(mems) > 1 else 0, 2),
            'exp_mean': round(statistics.mean(expanded), 1),
            'exp_std': round(statistics.stdev(expanded) if len(expanded) > 1 else 0, 1),
            'cost_mean': round(statistics.mean(costs), 4),
            'cost_std': round(statistics.stdev(costs) if len(costs) > 1 else 0, 4),
        })

    _print_batch_table(summaries)
    return summaries


def _print_batch_table(summaries):
    print("\nAlgo        Solved   Time(ms)        Mem(KB)        NodesExp      Cost")
    print("-" * 70)

    for s in summaries:
        if 'note' in s:
            print(f"{s['algo']:<10} {s['note']}")
            continue

        print(
            f"{s['algo']:<10} {s['n_solved']:>5}   "
            f"{s['time_mean']:>7.2f}±{s['time_std']:<5.2f}   "
            f"{s['mem_mean_kb']:>7.2f}±{s['mem_std_kb']:<5.2f}   "
            f"{s['exp_mean']:>7.1f}±{s['exp_std']:<5.1f}   "
            f"{s['cost_mean']:>7.2f}±{s['cost_std']:<5.2f}"
        )


# ── OPTIONAL: simple chart (extra credit easy win) ───────────────────────────

def plot_runtime(summaries):
    import matplotlib.pyplot as plt

    algos = [s['algo'] for s in summaries if 'time_mean' in s]
    times = [s['time_mean'] for s in summaries if 'time_mean' in s]

    plt.figure()
    plt.bar(algos, times)
    plt.title("Runtime Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (ms)")
    plt.show()