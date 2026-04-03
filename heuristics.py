import math

# GRID-BASED HEURISTICS

def manhattan(a, b):
    """
    Manhattan distance (4-direction grid)
    Admissible when movement is only up/down/left/right
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    """
    Euclidean distance (straight-line distance)
    Works for both grid and coordinate graphs
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def chebyshev(a, b):
    """
    Chebyshev distance (8-direction grid)
    Useful when diagonal movement is allowed
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


# CITY / COORDINATE HEURISTICS
def euclidean_coords(node_a, node_b, coords):
    """
    Euclidean distance using coordinate lookup
    node_a, node_b = node names (strings)
    coords = dict {node: (lat, lon)}
    """
    x1, y1 = coords[node_a]
    x2, y2 = coords[node_b]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def haversine(node_a, node_b, coords):
    """
    Haversine distance (real-world Earth distance)
    Good for city datasets with lat/lon
    """
    lat1, lon1 = coords[node_a]
    lat2, lon2 = coords[node_b]

    R = 6371  # Earth radius in km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


# WRAPPER 
def get_heuristic(name, coords=None):
    """
    Returns a heuristic function based on string name
    """

    if name == "manhattan":
        return manhattan

    elif name == "euclidean":
        return euclidean

    elif name == "chebyshev":
        return chebyshev

    elif name == "euclidean_coords":
        return lambda a, b: euclidean_coords(a, b, coords)

    elif name == "haversine":
        return lambda a, b: haversine(a, b, coords)

    else:
        raise ValueError(f"Unknown heuristic: {name}")