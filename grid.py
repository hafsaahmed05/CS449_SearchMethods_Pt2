from platform import node

import numpy as np
import random

'''
ChatGPT Prompt:
Based on the chat about the path search, how should I go about this project. 
Let's start with grid.py. What functions should I have in the grid class to create the 
grid environment?

The functions used in this class were given by ChatGPT and written in assistance with Claude. 

Code reused from Part 1.
'''

EMPTY    = 0
OBSTACLE = 1
START    = 2
GOAL     = 3

class GridEnvironment:
    def __init__(self, rows=10, cols=10, obstacle_pct=0.25, seed=None, connectivity=4):
        self.rows = rows
        self.cols = cols
        self.obstacle_pct = obstacle_pct  # should be between 0.20 and 0.30
        self.grid = None        # 2D numpy array of cell values
        self.obstacles = set()  # set of (r, c) tuples for O(1) lookup
        self.start = None       # (r, c) tuple
        self.goal  = None       # (r, c) tuple
        self.adjacency = {}     # {(r,c): [(r2,c2), ...]}

        # coords dict mirrors city_graph interface so heuristics work uniformly
        # For grid, coords[node] = (row, col) — euclidean uses row/col directly
        self.coords = {}

        # Seed the RNG if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.connectivity = connectivity

        # Retry loop: regenerate until start and goal are connected
        for _ in range(100):
            self._build_grid()
            if self.is_connected():
                break
        else:
            raise RuntimeError("Could not generate a connected grid after 100 tries. "
                               "Try reducing obstacle_pct.")

    # Top-level builder - calls phases in the correct order
    def _build_grid(self):
        """Full rebuild: clear state and run all three phases."""
        self.grid      = np.zeros((self.rows, self.cols), dtype=int)
        self.obstacles = set()
        self.adjacency = {}
        self.coords    = {}
        self.assign_obstacles()
        self.place_nodes()
        self.build_adjacency()

    # Phase 1 - Scatter obstacles across the grid
    def assign_obstacles(self):
        # Calculate number of obstacles to place on the grid based on user parameters
        total_cells  = self.rows * self.cols
        num_obstacles = int(total_cells * self.obstacle_pct)

        # Build a flat list of all positions, shuffle, take the first N
        all_positions = [(r, c) for r in range(self.rows)
                                  for c in range(self.cols)]
        random.shuffle(all_positions)

        for r, c in all_positions[:num_obstacles]:
            self.obstacles.add((r, c))
            self.grid[r][c] = OBSTACLE

    # Phase 2 - Place start and goal on valid (non-obstacle) cells
    def place_nodes(self):
        valid_cells = [(r, c) for r in range(self.rows)
                              for c in range(self.cols)
                              if (r, c) not in self.obstacles]

        if len(valid_cells) < 2:
            raise ValueError("Not enough open cells to place start and goal.")

        # Sample 2 distinct positions without replacement
        self.start, self.goal = random.sample(valid_cells, 2)

        # Those two random positions are used to determine the start and goal nodes
        self.grid[self.start[0]][self.start[1]] = START
        self.grid[self.goal[0]][self.goal[1]]   = GOAL

    '''
    Claude Prompt:
    Help give adjacency code based on the following GridEnvironment class and functions
    '''
    # Phase 3 - Build adjacency list (4-cardinal neighbors only)
    def get_neighbors(self, node):
        return self.adjacency.get(node, [])

    def build_adjacency(self):
        """
        Build adjacency list.
        4-connectivity: up/down/left/right (cardinal only).
        8-connectivity: cardinal + 4 diagonals.
        """
        if self.connectivity == 8:
            directions = [(-1,0),(1,0),(0,-1),(0,1),
                          (-1,-1),(-1,1),(1,-1),(1,1)]
        else:
            directions = [(-1,0),(1,0),(0,-1),(0,1)]

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.obstacles:
                    continue
                self.coords[(r, c)] = (r, c)   # grid coords = position
                neighbors = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if self.is_valid(nr, nc):
                        neighbors.append((nr, nc))
                self.adjacency[(r, c)] = neighbors

    # Helper - bounds check + not an obstacle
    def is_valid(self, r, c):
        in_bounds = (0 <= r < self.rows) and (0 <= c < self.cols)
        return in_bounds and (r, c) not in self.obstacles

    # Helper - BFS connectivity check (can start reach goal?)
    def is_connected(self):

        # If start or goal doesn't exist, there is no possible path
        if self.start is None or self.goal is None:
            return False

        visited = set()          # keeps track of explored nodes
        queue = [self.start]     # BFS starts from the start node

        # Explore the graph using Breadth-First Search
        while queue:
            node = queue.pop(0)

            # If the goal is reached, a path exists
            if node == self.goal:
                return True

            # Skip nodes that were already visited
            if node in visited:
                continue

            visited.add(node)

            # Add neighboring nodes to the queue for exploration
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # If all reachable nodes were checked and goal wasn't found
        return False

    # Debug helper - print to console
    def print_grid(self):
        symbols = {EMPTY: '.', OBSTACLE: '#', START: 'S', GOAL: 'G'}
        for r in range(self.rows):
            print(' '.join(symbols[self.grid[r][c]] for c in range(self.cols)))
        print(f"Start: {self.start}  |  Goal: {self.goal}  |  Connectivity: {self.connectivity}-conn")
