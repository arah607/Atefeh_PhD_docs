
import networkx as nx
from math import acos, degrees
import numpy as np


def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norms)
    return np.degrees(angle)

nodes_in_label = ['[-64.285, -156.097, -172.039]', '[-74.42, -168.794, -174.912]', '[-78.588, -174.22, -173.743]',
                  '[-123.028, -195.604, -209.888]', '[-107.654, -190.361, -194.845]', '[-106.124, -191.593, -196.018]', '[-112.298, -192.833, -200.023]',
                  '[-105.051, -190.802, -194.715]', '[-74.242, -183.626, -185.045]', '[-74.987, -169.306, -172.081]']
edges_in_label = [('[-64.285, -156.097, -172.039]', '[-74.987, -169.306, -172.081]'), ('[-74.42, -168.794, -174.912]', '[-74.987, -169.306, -172.081]'),
                  ('[-78.588, -174.22, -173.743]', '[-107.654, -190.361, -194.845]'), ('[-78.588, -174.22, -173.743]', '[-74.987, -169.306, -172.081]'),
                  ('[-78.588, -174.22, -173.743]', '[-74.242, -183.626, -185.045]'), ('[-123.028, -195.604, -209.888]', '[-107.654, -190.361, -194.845]'),
                  ('[-107.654, -190.361, -194.845]', '[-106.124, -191.593, -196.018]'), ('[-106.124, -191.593, -196.018]', '[-112.298, -192.833, -200.023]'),
                  ('[-106.124, -191.593, -196.018]', '[-105.051, -190.802, -194.715]')]
dag = nx.Graph()
dag.add_nodes_from(nodes_in_label)
dag.add_edges_from(edges_in_label)
# Find all simple paths in the DAG
all_paths = []
nodes = dag.nodes()




# node_choos = []
# for u in nodes:
#     if nx.degree(dag, u) == 1:
#         node_choos.append(u)

nodes = dag.nodes()
choosN = []
for N in nodes:
    if nx.degree(dag, N) == 1:
        choosN.append(N)

all_paths = set()
for node1 in choosN:
    for node2 in choosN:
        if node1 != node2:
            paths = nx.all_simple_paths(dag, source=node1, target=node2)
            for path in paths:
                if len(path) > 2:
                    angles = []
                    for i in range(len(path) - 2):
                        vector1 = np.array(eval(path[i]))
                        vector2 = np.array(eval(path[i + 1]))
                        vector3 = np.array(eval(path[i + 2]))
                        angle = calculate_angle(vector2 - vector1, vector3 - vector2)
                        angles.append(angle)
                    if all(angle < 45 for angle in angles):
                        all_paths.add(tuple(path))
# def find_all_paths(graph, source, target, path=[]):
#     path = path + [source]
#     if source == target:
#         return [path]
#     if source not in graph:
#         return []
#     paths = []
#     for neighbor in graph[source]:
#         if neighbor not in path:
#             new_paths = nx.all_simple_paths(graph, neighbor, target, path)
#             paths.extend(new_paths)
#     return paths
# source = '[-52.507, -146.084, -165.937]'
# target = '[-70.664, -93.07, -176.556]'
#
# all_paths = find_all_paths(dag, source, target)