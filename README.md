# CS449 — AI Search Methods · Part 2
### Heuristic Search + Benchmarking Visualizer

An interactive visualizer for comparing uninformed and informed search algorithms on two environments: a randomly generated grid world and a real weighted city graph of southern Kansas.

---

## Features

- **5 Search Algorithms** — A*, Greedy Best-First, BFS, DFS, IDDFS
- **2 Graph Modes** — Random grid world or 45-city Kansas road network
- **3 Grid Heuristics** — Manhattan, Euclidean, Chebyshev
- **3 City Heuristics** — Euclidean (coordinate lookup), Haversine, Euclidean
- **Step-by-step animation** with adjustable speed
- **Live metrics panel** — nodes expanded, frontier size, g(n), h(n), f(n), path cost, depth
- **Search tree** visualization alongside the main graph
- **Frontier queue** display with f-values, labeled by `(row, col)`
- **Hover/click tooltips** — after search completes, hover any node to see g(n), h(n), f(n), and parent; click to pin
- **Quick presets** — Easy / Medium / Hard buttons instantly configure and run the grid
- **Benchmark tab** — in-GUI batch comparison with bar charts for runtime, memory, and nodes expanded

---

## Project Structure

```
CS449_SearchMethods_Pt2/
├── main.py                  # Entry point — launches the GUI
├── requirements.txt         # Python dependencies
├── README.md
├── .gitignore
│
├── core/                    # Search logic
│   ├── __init__.py
│   ├── search.py            # BFS, DFS, IDDFS, Greedy, A*
│   ├── heuristics.py        # Manhattan, Euclidean, Chebyshev, Haversine
│   ├── grid.py              # Random grid environment
│   └── graph_loader.py      # City graph loader (CSV + adjacency)
│
├── ui/                      # Interface
│   ├── __init__.py
│   ├── visualization.py     # Main single-window visualizer
│   └── launcher.py          # Two-page setup launcher
│
├── benchmark/               # Profiling and comparison
│   ├── __init__.py
│   └── benchmark.py         # Batch runner, metrics, charts
│
└── data/                    # Input data
    ├── coordinates.csv      # Lat/lon for 45 Kansas cities
    └── Adjacencies.txt      # City adjacency pairs (bidirectional)
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

```bash
python main.py
```

Optional flags for custom data files:
```bash
python main.py --coord_file coordinates.csv --adj_file Adjacencies.txt
```

---

## Usage

### Grid World Mode
1. Select **Grid World** as the graph mode
2. Set rows, cols (5–30), obstacle density (20–30%), and connectivity (4-way or 8-way)
3. Pick an algorithm and heuristic
4. Click **Run** — start and goal are placed randomly
5. Or use **Easy / Medium / Hard** quick preset buttons to auto-configure and run instantly
6. Use **Pause / Resume** to step through, **Reset** to clear, **Close** to exit

### City Graph Mode
1. Select **City Graph**
2. Choose a start city and goal city from the dropdowns (45 Kansas cities)
3. Pick a layout (Geographic = real lat/lon positions, Spring = force-directed)
4. Select an algorithm and heuristic
5. Click **Run**

### Node Tooltips (after search completes)
- **Hover** over any grid cell or city node to see g(n), h(n), f(n), and parent
- **Click** to pin the tooltip so it stays visible while you move the mouse
- **Click again** to unpin and dismiss
- Tooltips are inactive during animation to avoid overhead

### Reading the Display
- **Grid axes** show `row` (top→bottom) and `col` (left→right), 0-indexed — matches `(row, col)` tuples in the queue
- **Queue widget** lists frontier nodes sorted by f-value
- **Search tree** shows the exploration tree with g/h/f annotation on the current node
- **Metrics bar** shows live stats for the current search step

### Benchmark Tab
1. Click **⧖ Benchmark** to switch to benchmark mode
2. **Grid mode** — choose Easy+Med+Hard (runs all 3 complexity settings) or Custom (set your own grid size and runs)
3. **City mode** — uses the Start/Goal cities selected above; runs all 5 algorithms on that pair
4. Click **▶ Run Benchmark** — results appear as a 3-panel bar chart and are saved to `benchmark_results.png`
5. Click **⬡ Search** to return to normal search mode

---

## Algorithms

| Algorithm | Complete | Optimal | Notes |
|-----------|----------|---------|-------|
| BFS | Yes | Yes (unit cost) | Explores level by level |
| DFS | Yes | No | Low memory, dives deep first |
| IDDFS | Yes | Yes (unit cost) | DFS memory + BFS optimality |
| Greedy | No | No | Fast, guided by h(n) only |
| A* | Yes | Yes | Optimal with admissible heuristic |

---

## Heuristics

**Grid:**
- `manhattan` — admissible for 4-way movement
- `euclidean` — straight-line distance, admissible for 8-way
- `chebyshev` — admissible for 8-way (diagonal) movement

**City Graph:**
- `euclidean_coords` — straight-line distance using lat/lon coordinates
- `haversine` — great-circle distance (real Earth distance in km)
- `euclidean` — automatically uses coordinate lookup in city mode

---

## City Graph Data

45 cities in southern Kansas with latitude/longitude coordinates (`coordinates.csv`) and road adjacencies (`Adjacencies.txt`). Adjacencies are bidirectional — if A↔B is listed, travel is supported in both directions. Edge weights are computed as Euclidean distance between coordinates.

---

## Benchmarking

The benchmark tab runs all 5 algorithms across complexity settings and produces:

- Comparison table with mean ± std for time, memory, nodes expanded, and path cost
- Optimality marker (✓) identifying which algorithms found the lowest-cost path
- Seeds recorded in output for reproducibility
- 3-panel bar chart saved to `benchmark_results.png`

**Complexity settings (grid):**

| Setting | Grid Size | Obstacle Density |
|---------|-----------|-----------------|
| Easy    | 8×8       | 20%             |
| Medium  | 15×15     | 25%             |
| Hard    | 25×25     | 30%             |

To run programmatically:
```python
from benchmark import run_complexity_suite
run_complexity_suite()
```

---

## AI Assistance

Grid environment structure and adjacency builder were developed with assistance from ChatGPT and Claude. Prompt sequences are documented inline in each source file.