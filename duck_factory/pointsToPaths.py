import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt

from model3d import Mesh

import random

def create_graph(points, max_distance):
    G = nx.Graph()
    for i, (x, y, z) in enumerate(points):
        G.add_node(i, pos=(x, y, z))
    
    dist_matrix = distance_matrix(points, points)
    num_points = len(points)
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if dist_matrix[i, j] <= max_distance:
                G.add_edge(i, j, weight=dist_matrix[i, j])
    
    return G

def find_connected_components(G):
    return [list(component) for component in nx.connected_components(G)]

def solve_path(points, max_distance):
    """Finds paths ensuring consecutive points are within max_distance."""
    num_points = len(points)
    if num_points <= 1:
        return [points] if num_points == 1 else []
    
    points = np.array(points)
    unvisited = set(range(num_points))
    
    paths = []
    
    while unvisited:
        path = []
        start = unvisited.pop()
        path.append(tuple(points[start]))
        
        while unvisited:
            last = path[-1]
            nearest = min(unvisited, key=lambda i: np.linalg.norm(points[i] - last))
            
            # Check distance constraint
            if np.linalg.norm(points[nearest] - last) > max_distance:
                break  # Stop this path and start a new one
            
            unvisited.remove(nearest)
            path.append(tuple(points[nearest]))
        
        paths.append(path) 
    
    return paths

def find_paths(points, max_distance):
    G = create_graph(points, max_distance)
    components = find_connected_components(G)
    
    paths = []
    for component in components:
        component_points = [points[i] for i in component]
        component_paths = solve_path(component_points, max_distance)
        paths.extend(component_paths)
    
    return paths

def plot_paths(points, paths):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    points = np.array(points)
    
    for i, path in enumerate(paths):
        path = np.array(path)
        if len(path) > 1:
            ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='red', marker='o')
            ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', linestyle='-', label=f'Path {i+1}')
        else:
            ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='blue', marker='x')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


# Example usage

def main():
    mesh = Mesh("duck_factory/model.obj", "duck_factory/Génère_moi_un_canar_0219074804_texture.png")

    points = mesh.get_point_cloud()
    random.shuffle(points)
    points = points[:100]

    print(f"Number of points: {len(points)}")
    points_all = [point[0] for point in points]

    paths = find_paths(points_all, 0.1)
    print(f"Number of paths (length > 1) : {len([path for path in paths if len(path) > 1])}")

    print(f"Number of paths (length <= 1) : {len([path for path in paths if len(path) <= 1])}")

    # for i, path in enumerate(paths):
        # print(f"Path {i+1}: {path}")

    plot_paths(points_all, paths)

if __name__ == "__main__":
    main()