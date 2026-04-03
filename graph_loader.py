import csv
import math

class WeightedGraph:
    def __init__(self):
        self.adjacency = {}   # {node: [(neighbor, weight)]}
        self.coords = {}      # {node: (x, y)}

    def add_node(self, name, x, y):
        self.coords[name] = (float(x), float(y))
        if name not in self.adjacency:
            self.adjacency[name] = []

    def add_edge(self, a, b, weight):
        self.adjacency.setdefault(a, []).append((b, weight))
        self.adjacency.setdefault(b, []).append((a, weight))

    def compute_distance(self, a, b):
        x1, y1 = self.coords[a]
        x2, y2 = self.coords[b]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def edge_weight(self, a, b):
        for neighbor, w in self.adjacency.get(a, []):
            if neighbor == b:
                return w
        return float('inf')


def load_city_graph(coord_file, adj_file):
    graph = WeightedGraph()

    # ── Load coordinates.csv ───────────────────────────────────────────
    with open(coord_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            name = row[0].strip()
            try:
                x, y = float(row[1].strip()), float(row[2].strip())
            except ValueError:
                continue
            graph.add_node(name, x, y)

    # ── Load adjacency list (space-separated, Windows line endings ok) ──
    with open(adj_file) as f:
        for line in f:
            # Strip \r\n and split on whitespace
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a not in graph.coords or b not in graph.coords:
                continue
            weight = graph.compute_distance(a, b)
            graph.add_edge(a, b, weight)

    return graph