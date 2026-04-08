from collections import deque
import heapq

# Search algorithm implementations referenced from GeeksforGeeks (geeksforgeeks.org)
# and adapted to use Python generators for step-by-step visualization.

# BFS: https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
# DFS: https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
# IDDFS: https://www.geeksforgeeks.org/iterative-deepening-searchids-iterative-deepening-depth-first-searchiddfs/
# Greedy Best-First: https://www.geeksforgeeks.org/greedy-best-first-search-algorithm/
# A*: https://www.geeksforgeeks.org/a-search-algorithm/

# ── Path reconstruction ───────────────────────────────────────────────────────
def reconstruct_path(parent, start, goal):
    path    = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    path.reverse()
    return path if path and path[0] == start else []


# ── BFS ───────────────────────────────────────────────────────────────────────
def bfs(start, goal, graph):
    frontier = deque([start])
    visited  = set()
    parent   = {start: None}

    while frontier:
        current = frontier.popleft()
        if current in visited:
            continue
        visited.add(current)

        yield {
            'current'  : current,
            'frontier' : list(frontier),
            'visited'  : set(visited),
            'parent'   : dict(parent),
            'found'    : current == goal,
            'algo'     : 'bfs',
            'g'        : {},
            'h'        : {},
            'f'        : {},
        }

        if current == goal:
            return

        for neighbor in graph.get(current, []):
            nb = neighbor[0] if (isinstance(neighbor, tuple) and isinstance(neighbor[0], str)) else neighbor
            if nb not in visited and nb not in parent:
                parent[nb] = current
                frontier.append(nb)


# ── DFS ───────────────────────────────────────────────────────────────────────
def dfs(start, goal, graph):
    stack   = [start]
    visited = set()
    parent  = {start: None}

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)

        yield {
            'current'  : current,
            'frontier' : list(stack),
            'visited'  : set(visited),
            'parent'   : dict(parent),
            'found'    : current == goal,
            'algo'     : 'dfs',
            'g'        : {},
            'h'        : {},
            'f'        : {},
        }

        if current == goal:
            return

        for neighbor in reversed(graph.get(current, [])):
            nb = neighbor[0] if (isinstance(neighbor, tuple) and isinstance(neighbor[0], str)) else neighbor
            if nb not in visited and nb not in parent:
                parent[nb] = current
                stack.append(nb)


# ── IDDFS ─────────────────────────────────────────────────────────────────────
def iddfs(start, goal, graph, max_depth=100):

    def dls(node, goal, depth, parent, visited, g_cost):
        yield {
            'current'  : node,
            'frontier' : [],
            'visited'  : set(visited),
            'parent'   : dict(parent),
            'found'    : node == goal,
            'algo'     : 'iddfs',
            'g'        : dict(g_cost),
            'h'        : {},
            'f'        : {},
        }
        if node == goal:
            return True
        if depth <= 0:
            return False
        visited.add(node)
        for neighbor in graph.get(node, []):
            nb = neighbor[0] if (isinstance(neighbor, tuple) and isinstance(neighbor[0], str)) else neighbor
            cost = neighbor[1] if (isinstance(neighbor, tuple) and isinstance(neighbor[0], str)) else 1.0
            if nb not in visited:
                parent[nb] = node
                g_cost[nb] = g_cost.get(node, 0) + cost
                found = yield from dls(nb, goal, depth - 1, parent, visited, g_cost)
                if found:
                    return True
        return False

    for depth in range(max_depth + 1):
        visited = set()
        parent  = {start: None}
        g_cost  = {start: 0.0}
        found   = yield from dls(start, goal, depth, parent, visited, g_cost)
        if found:
            return


# ── Greedy Best-First ─────────────────────────────────────────────────────────
def greedy_best_first(start, goal, graph, heuristic):
    h_start  = heuristic(start, goal)
    open_list = []
    heapq.heappush(open_list, (h_start, start))

    visited  = set()
    parent   = {start: None}
    h_vals   = {start: h_start}

    while open_list:
        h, current = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)

        yield {
            'current'  : current,
            'frontier' : [n for _, n in open_list],
            'visited'  : set(visited),
            'parent'   : dict(parent),
            'found'    : current == goal,
            'algo'     : 'greedy',
            'g'        : {},
            'h'        : dict(h_vals),
            'f'        : dict(h_vals),   # greedy: f = h
        }

        if current == goal:
            return

        for neighbor in graph.get(current, []):
            if isinstance(neighbor, tuple) and isinstance(neighbor[0], str):
                nb = neighbor[0]
            else:
                nb = neighbor

            if nb not in visited:
                parent[nb]  = current
                h_val       = heuristic(nb, goal)
                h_vals[nb]  = h_val
                heapq.heappush(open_list, (h_val, nb))


# ── A* ────────────────────────────────────────────────────────────────────────
def astar(start, goal, graph, heuristic):
    h_start   = heuristic(start, goal)
    open_list = []
    heapq.heappush(open_list, (h_start, start))

    g_cost   = {start: 0.0}
    h_vals   = {start: h_start}
    f_vals   = {start: h_start}
    parent   = {start: None}
    visited  = set()

    while open_list:
        f, current = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)

        yield {
            'current'  : current,
            'frontier' : [n for _, n in open_list],
            'visited'  : set(visited),
            'parent'   : dict(parent),
            'found'    : current == goal,
            'algo'     : 'astar',
            'g'        : dict(g_cost),
            'h'        : dict(h_vals),
            'f'        : dict(f_vals),
        }

        if current == goal:
            return

        for neighbor in graph.get(current, []):
            # Distinguish city graph edge (str, weight) from grid node (int, int)
            if isinstance(neighbor, tuple) and isinstance(neighbor[0], str):
                nb, cost = neighbor
            else:
                nb   = neighbor
                cost = 1.0

            tentative_g = g_cost[current] + cost

            if nb not in g_cost or tentative_g < g_cost[nb]:
                g_cost[nb] = tentative_g
                h_val      = heuristic(nb, goal)
                f_val      = tentative_g + h_val

                h_vals[nb] = h_val
                f_vals[nb] = f_val
                parent[nb] = current

                heapq.heappush(open_list, (f_val, nb))


# ── Registry ──────────────────────────────────────────────────────────────────
ALGORITHMS = {
    'bfs'   : bfs,
    'dfs'   : dfs,
    'iddfs' : iddfs,
    'greedy': greedy_best_first,
    'astar' : astar,
}

def get_algorithm(name):
    return ALGORITHMS.get(name, bfs)