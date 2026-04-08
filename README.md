# CS449 — AI Search Methods · Part 2
### Heuristic Search + Benchmarking Visualizer

An interactive visualizer for comparing uninformed and informed search algorithms on two environments: a randomly generated grid world and a real weighted city graph of southern Kansas.

---

## Features

- **5 Search Algorithms** — A\*, Greedy Best-First, BFS, DFS, IDDFS
- **2 Graph Modes** — Random grid world or 45-city Kansas road network
- **3 Grid Heuristics** — Manhattan, Euclidean, Chebyshev
- **3 City Heuristics** — Euclidean (coordinate lookup), Haversine, Euclidean
- **Step-by-step animation** with adjustable speed
- **Live metrics panel** — nodes expanded, frontier size, g(n), h(n), f(n), path cost, depth
- **Search tree** visualization alongside the main graph
- **Frontier queue** display with f-values, labeled by `(row, col)`
- **Benchmarking mode** — batch comparison of all algorithms with timing, memory, and cost stats

---

## Project Structure

```
├── main.py            # Entry point — launches GUI or benchmark mode
├── visualization.py   # Single-window Tkinter + matplotlib visualizer
├── launcher.py        # Two-page setup launcher (alternate entry)
├── search.py          # BFS, DFS, IDDFS, Greedy, A* (generator-based)
├── heuristics.py      # Manhattan, Euclidean, Chebyshev, Haversine, euclidean_coords
├── grid.py            # Random grid environment with obstacle placement
├── graph_loader.py    # Loads city graph from CSV + adjacency file
├── benchmark.py       # Profiler: time, memory, nodes expanded, path cost
├── coordinates.csv    # Lat/lon for 45 Kansas cities
└── Adjacencies.txt    # City adjacency pairs (bidirectional)
```

---

## Requirements

```
numpy
matplotlib
networkx
tkinter  # built into Python; on Linux may need: sudo apt install python3-tk
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running

### Standard GUI
```bash
python main.py
```

### With custom data files
```bash
python main.py --coord_file coordinates.csv --adj_file Adjacencies.txt
```

### Benchmark mode (no GUI, prints results to terminal)
```bash
python main.py --benchmark
```

---

## Usage

### Grid World Mode
1. Select **Grid World** as the graph mode
2. Set rows, cols (5–30), obstacle density (20–30%), and connectivity (4-way or 8-way)
3. Pick an algorithm and heuristic
4. Click **Run** — start and goal are placed randomly
5. Use **Pause / Resume** to step through, **Reset** to clear, **Close** to exit

### City Graph Mode
1. Select **City Graph**
2. Choose a start city and goal city from the dropdown (45 Kansas cities)
3. Pick a layout (Geographic = real lat/lon positions, Spring = force-directed)
4. Select an algorithm and heuristic
5. Click **Run**

### Reading the Display
- **Grid labels**: axes show `row` (top→bottom) and `col` (left→right), 0-indexed — matches the `(row, col)` tuples shown in the queue
- **Queue widget**: lists frontier nodes by f-value with `(row, col)` coordinates
- **Search tree**: shows the exploration tree with node labels and g/h/f annotation on the current node
- **Metrics bar**: live stats for the current search step

---

## Algorithms

| Algorithm | Complete | Optimal | Notes |
|-----------|----------|---------|-------|
| BFS | Yes | Yes (unit cost) | Explores level by level |
| DFS | Yes | No | Low memory, dives deep first |
| IDDFS | Yes | Yes (unit cost) | DFS memory + BFS optimality |
| Greedy | No | No | Fast, guided by h(n) only |
| A\* | Yes | Yes | Optimal with admissible heuristic |

---

## Heuristics

**Grid:**
- `manhattan` — admissible for 4-way movement
- `euclidean` — straight-line distance, admissible for 8-way
- `chebyshev` — admissible for 8-way (diagonal) movement

**City Graph:**
- `euclidean_coords` — straight-line distance using lat/lon coordinates
- `haversine` — great-circle distance (real Earth distance in km)
- `euclidean` — falls back to coordinate lookup automatically in city mode

---

## City Graph Data

45 cities in southern Kansas with latitude/longitude coordinates (`coordinates.csv`) and road adjacencies (`Adjacencies.txt`). Adjacencies are bidirectional — if A↔B is listed, the graph supports travel in both directions. Edge weights are computed as Euclidean distance between coordinates.

---

## Benchmarking

Run `python main.py --benchmark` to compare all 5 algorithms across multiple random grid seeds. Output includes:

- Wall-clock time (ms)
- Peak memory (KB)  
- Nodes expanded
- Path cost
- Branching factor (avg/max)

To benchmark programmatically:
```python
from benchmark import batch_compare
from grid import GridEnvironment

def env_factory(seed):
    return GridEnvironment(rows=10, cols=10, obstacle_pct=0.25, connectivity=4, seed=seed)

batch_compare(env_factory, n_runs=10)
```

---

## AI Assistance

Grid environment structure and adjacency builder were developed with assistance from ChatGPT and Claude. Prompt sequences are documented inline in `grid.py`.