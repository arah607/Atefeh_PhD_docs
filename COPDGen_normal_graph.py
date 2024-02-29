import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.neighbors import KDTree
import collections
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import stats
import statistics
from math import pi
# import vtk_nifit_render
import scipy.ndimage
from skimage.morphology import ball,cube
import time
import numpy.linalg as LA

class Vertex:
    """ Class to define vertices of the graph"""
    def __init__(self, point, degree=0, edges=None):
        # self.point = np.asarray(point)
        self.point = point
        self.degree = degree
        self.edges = []
        self.visited = False
        if edges is not None:
            self.edges = edges

    def __str__(self):
        return str(self.point)

#Class to define edges of the graph
class Edge:
    """Class to define edges of the graph"""
    def __init__(self, start, end=None, pixels=None):
        self.start = start
        self.end = end
        self.pixels = []
        if pixels is not None:
            self.pixels = pixels
        self.visited = False
def buildTree_tree(coor, start=None):
    """Builds graph from centerline based on pixel connectivity. Gets every white pixel
    and converts it to black pixels once its added to the graph. Nodes are created using
    class Vertex and added to graph, similarly edges are created and added"""
    # copy image since we set visited pixels to black

    # a list of all graphs extracted from the skeleton
    graphs = []

    i=0
    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    G = nx.Graph()
    # G = G1.copy()
    i = 0
    nWhitePixels = len(coor)
    #coor = round(coor,2)
    X = []
    delpoint = []
    record = []
    time_total = 0
    while nWhitePixels != blackedPixels:

        start_time = time.time()
        new_start = True
        disconect_tree = True
        # point_array = np.array(coor)
        # b = []
        # for h in range(len(point_array)):
        #     b.append(point_array[h])


        while disconect_tree:
            Find_neigh = False

            point_array = np.array(coor)
            # b = np.array(b)

            point = coor[0]
            X.append(point)

            # point = b[i]

            # queue = collections.deque()
            # for h in range(len(point_array)):
            #     queue.append(point_array[h])
            if new_start:
                if point not in X:
                    X.append(point)
                new_start=False
            tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)
            # tree = KDTree(b, leaf_size=b.shape[0] + 1)

            count_neigh = 0
            if len (point_array) >2:
                distances, ind = tree.query([point], k=2)


            elif len(point_array) == 2:
                distances, ind = tree.query([point], k=2)


            else:
                distances, ind = tree.query([point], k=1)
                blackedPixels += 1
                break


            for j in range(1, len(distances[0])):

                if len(delpoint)!=0:
                    dic = {}
                    for c in range(len(delpoint)):
                        dist = np.linalg.norm(np.array(point) - np.array(delpoint[c]))
                        dic[dist] = delpoint[c]
                    per_point = dic[min(dic.keys())]

                    if not G.has_edge(str(point), str(per_point)) and min(dic.keys()) < 0.8 :  #1.5033 #
                        if not G.has_node(point):
                            point = Vertex(point)
                            G.add_node(str(point), pos=point.point)
                            point = list(point.point)
                            if not nx.has_path(G, str((point)), str(per_point)):
                                currV = Vertex(point)
                                pervious_point = Vertex(per_point)
                                G.add_node(str(currV.point), pos=currV.point)
                                G.add_node(str(pervious_point.point), pos=pervious_point.point)
                                G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))


                #X.append(point)
                if distances[0][j] <= 0.7:

                    nb_closest = coor[ind[0][j]]



                    if nb_closest not in X:
                        X.append(nb_closest)
                        # if point not in X:
                        #     X.append(point)

                        newV = Vertex(nb_closest)
                        currV = Vertex(point)
                        G.add_node(str(currV.point), pos=currV.point)
                        G.add_node(str(newV.point), pos=newV.point)
                        G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
                        count_neigh = count_neigh + 1
                        # coor.remove(currV.point)
                        #blackedPixels += 1
                        # network_plot_3D(G, angle=0)
                    elif nb_closest in X:
                        # pervious_point = coor[ind[0][j]]
                        # pervious_point = Vertex(pervious_point)
                        # currV = Vertex(point)
                        if str(point) not in G.nodes:
                            point = Vertex(point)
                            G.add_node(str(point.point), pos = point.point)
                            point = list(point.point)
                            if not G.has_edge(str(point),str(coor[ind[0][j]])) and not nx.has_path(G, str((point)), str(coor[ind[0][j]])):
                                currV = Vertex(point)
                                pervious_point = Vertex(coor[ind[0][j]])
                                G.add_node(str(currV.point), pos=currV.point)
                                G.add_node(str(pervious_point.point), pos=pervious_point.point)
                                G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))
                        # coor.remove(currV.point)
                                #network_plot_3D(G, angle=0)
                        # point_array = np.delete(point_array, i, axis=0)

            if count_neigh == 0 and distances[0][1]<2:
                disconect_tree = False
                #network_plot_3D(G, angle=0)

                for j in range(1, len(distances[0])):
                    nb_closest = coor[ind[0][j]]
                    if nb_closest not in X:
                        X.append(nb_closest)
                        # if point not in X:
                        #     X.append(point)
                        newV = Vertex(nb_closest)
                        #blackedPixels += 1
                        currV = Vertex(point)
                        G.add_node(str(currV.point), pos=currV.point)
                        G.add_node(str(newV.point), pos=newV.point)
                        G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
                        #blackedPixels += 1
                        Find_neigh=True
                        # coor.remove(currV.point)
                        #network_plot_3D(G, angle=0)
                        break

                    elif nb_closest in X:
                        # currV = Vertex(point)
                        # pervious_point = coor[ind[0][j]]
                        # pervious_point = Vertex(pervious_point)
                        if str(point) not in G.nodes:
                            point = Vertex(point)
                            G.add_node(str(point.point), pos=point.point)
                            point = list(point.point)
                            if not G.has_edge(str(point),str(coor[ind[0][j]])) and not nx.has_path(G, str((point)), str(coor[ind[0][j]])):
                                currV = Vertex(point)
                                pervious_point = Vertex(coor[ind[0][j]])
                                G.add_node(str(currV.point), pos=currV.point)
                                G.add_node(str(pervious_point.point), pos=pervious_point.point)
                                G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))
                                # coor.remove(currV.point)
                                #network_plot_3D(G, angle=0)

                            break
                        # point_array = np.delete(point_array, i, axis=0)

                if Find_neigh == False:
                    for j in range(1, len(distances[0])):
                        currV = Vertex(point)
                        nb_closest = coor[ind[0][j]]
                        nb_closest_Vertex = Vertex(nb_closest)
                        if not G.has_edge(str(currV.point),str(nb_closest_Vertex.point))  and not nx.has_path(G, str(currV.point),str(nb_closest_Vertex.point)): #and nb_closest_Vertex.point > currV.point.point
                            X.append(nb_closest)
                            newV = Vertex(nb_closest)

                            currV = Vertex(point)
                            G.add_node(str(currV.point), pos=currV.point)
                            G.add_node(str(newV.point), pos=newV.point)
                            G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))

                            #blackedPixels += 1
                            #network_plot_3D(G, angle=0)
                            break
            elif count_neigh == 0 and distances[0][1]>2: #remove the edge from ends points
                i = i + 1
                blackedPixels += 1
                coor.remove(point)
                end_time = time.time()
                execution_time = end_time - start_time
                record.append(execution_time)
                time_total = time_total + execution_time
                break


            i = i + 1
            blackedPixels += 1

            if len(coor) > 1:
                delpoint.append(currV.point)
                coor.remove(currV.point) # remove the currv from the list and find the other neigbours
                end_time = time.time()
                execution_time = end_time - start_time
                record.append(execution_time)
                time_total = time_total + execution_time
            else:
                end_time = time.time()
                execution_time = end_time - start_time
                record.append(execution_time)
                time_total = time_total + execution_time
                break



    return G


def buildTree_all_branch(coor1, coor, trimmedGraph, endNodes_main_artery, start=None):
    X = []

    a=[]
    time_total=0
    G = trimmedGraph.copy()
    coor = np.array(coor) #coordinates for maine pul artery
    coor1 = np.array(coor1)#whole filter coordinates
    for j in range(len(coor)):
        if coor[j] in coor1[:]:
            b = np.where(np.all(coor1[:] == coor[j], axis=1))
            b = list(b)
            a.append(b)
    k = []
    for y in range(len(a)):
        k.append(a[y][0][0])

    arr = np.delete(coor1, k, axis=0)
    arr = arr.tolist()

    from ast import literal_eval
    for l in range(len(endNodes_main_artery)):
        endNodes_main_artery[l] = literal_eval(endNodes_main_artery[l])
        arr.append(endNodes_main_artery[l])

    arr.sort(key=lambda x: x[0])

    point_array = np.array(arr)
    point_array_end_artery = np.array(endNodes_main_artery)

    for h in range(len(endNodes_main_artery)):
        # point_array = np.unique(point_array, axis=0)
        point_indx = np.where(np.all(point_array[:] == point_array_end_artery[h], axis=1))
        point = point_array[point_indx][0]
        point = list(point)
        X.append(point)

        tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)
        count_neigh = 0
        distances, ind = tree.query([point], k=3)
        for j in range(1, len(distances[0])):
            if distances[0][j] <= 1:
                nb_closest = arr[ind[0][j]]
                if nb_closest not in X:
                    X.append(nb_closest)

                    newV = Vertex(nb_closest)
                    currV = Vertex(point)
                    G.add_node(str(currV.point), pos=currV.point)
                    G.add_node(str(newV.point), pos=newV.point)
                    G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
                    count_neigh = count_neigh + 1
                elif nb_closest in X:
                    # pervious_point = coor[ind[0][j]]
                    # pervious_point = Vertex(pervious_point)
                    # currV = Vertex(point)
                    if not G.has_edge(str(point), str(arr[ind[0][j]])):
                        currV = Vertex(point)
                        pervious_point = Vertex(coor1[ind[0][j]])
                        G.add_node(str(currV.point), pos=currV.point)
                        G.add_node(str(pervious_point.point), pos=pervious_point.point)
                        G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))

                arr.remove(point)
                point_array = np.array(arr)
                point = arr[ind[0][j]]

                tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)
                distances1, ind = tree.query([point], k=3)
                for j in range(1, len(distances1[0])):
                    if distances1[0][j] <= 2:
                        nb_closest = arr[ind[0][j]]
                        if nb_closest not in X:
                            X.append(nb_closest)

                            newV = Vertex(nb_closest)
                            currV = Vertex(point)
                            G.add_node(str(currV.point), pos=currV.point)
                            G.add_node(str(newV.point), pos=newV.point)
                            G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
                            count_neigh = count_neigh + 1
                        elif nb_closest in X:
                            # pervious_point = coor[ind[0][j]]
                            # pervious_point = Vertex(pervious_point)
                            # currV = Vertex(point)
                            if not G.has_edge(str(point), str(arr[ind[0][j]])):
                                currV = Vertex(point)
                                pervious_point = Vertex(coor1[ind[0][j]])
                                G.add_node(str(currV.point), pos=currV.point)
                                G.add_node(str(pervious_point.point), pos=pervious_point.point)
                                G.add_edge(str(currV.point), str(pervious_point.point),
                                           object=Edge(currV, pervious_point))
                    if distances1[0][j] > 2:
                        break

            if distances[0][j] > 1:
                break

    return G

# def buildTree_all_branch1(coor1, coor, trimmedGraph, endNodes_main_artery, start=None):
#     X = []
#
#     a=[]
#
#     coor = np.array(coor) #coordinates for maine pul artery
#     coor1 = np.array(coor1)#whole filter coordinates
#     for j in range(len(coor)):
#         check_exist=np.where(np.all(coor1[:] == coor[j], axis=1))
#         check_exist=list(check_exist)
#         if check_exist[0]:
#             #b =
#             #b = list(b)
#             a.append(check_exist)
#     k = []
#     for y in range(len(a)):
#             k.append(a[y][0][0])
#
#     arr = np.delete(coor1, k, axis=0)
#     arr = arr.tolist()
#
#     end = []
#     from ast import literal_eval
#     for l in range(len(endNodes_main_artery)):
#         endNodes_main_artery[l] = literal_eval(endNodes_main_artery[l])
#         end.append(endNodes_main_artery[l])
#         arr.append(endNodes_main_artery[l])
#     arr.sort(key=lambda x: x[0])
#
#     # coorNew = np.array(arr)
#     point_array_end_artery = np.array(endNodes_main_artery)
#
#     graphs = []
#
#     i = 0
#     blackedPixels = 0
#     # we build our graph as long as we have not blacked all white pixels!
#     # G = nx.Graph()
#     G = trimmedGraph.copy()
#     i = 0
#     nWhitePixels = len(arr)
#     # coor = round(coor,2)
#     X = []
#     delpoint = []
#     while nWhitePixels != blackedPixels:
#
#         new_start = True
#         disconect_tree = True
#         # point_array = np.array(coor)
#         # b = []
#         # for h in range(len(point_array)):
#         #     b.append(point_array[h])
#
#         while disconect_tree:
#             Find_neigh = False
#
#             point_array = np.array(arr)
#             # b = np.array(b)
#
#             point = arr[0]
#
#             # point = b[i]
#
#             # queue = collections.deque()
#             # for h in range(len(point_array)):
#             #     queue.append(point_array[h])
#             if new_start:
#                 X.append(point)
#                 new_start = False
#             tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)
#             # tree = KDTree(b, leaf_size=b.shape[0] + 1)
#
#             count_neigh = 0
#             if len(point_array) > 2:
#                 distances, ind = tree.query([point], k=3)
#
#
#             elif len(point_array) == 2:
#                 distances, ind = tree.query([point], k=2)
#
#
#             else:
#                 distances, ind = tree.query([point], k=1)
#                 blackedPixels += 1
#                 break
#
#             # if distances[0][1] > 8: # seprate the small trees
#             #     blackedPixels += 1
#             #     coor.remove(point)
#             #     break
#
#             for j in range(1, len(distances[0])):
#
#                 # if len(delpoint) != 0:
#                 #     dic = {}
#                 #     for c in range(len(delpoint)):
#                 #         dist = np.linalg.norm(np.array(point) - np.array(delpoint[c]))
#                 #         dic[dist] = delpoint[c]
#                 #     per_point = dic[min(dic.keys())]
#                 #
#                 #     if not G.has_edge(str(point), str(per_point)) and min(dic.keys()) < 0.8:
#                 #         currV = Vertex(point)
#                 #         pervious_point = Vertex(per_point)
#                 #         G.add_node(str(currV.point), pos=currV.point)
#                 #         G.add_node(str(pervious_point.point), pos=pervious_point.point)
#                 #         G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))
#
#                 # X.append(point)
#
#
#                 if distances[0][j] <= 0.625:
#
#                     nb_closest = arr[ind[0][j]]
#                     if nb_closest not in X:
#                         X.append(nb_closest)
#
#                         newV = Vertex(nb_closest)
#                         currV = Vertex(point)
#                         G.add_node(str(currV.point), pos=currV.point)
#                         G.add_node(str(newV.point), pos=newV.point)
#                         G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
#                         count_neigh = count_neigh + 1  # coor.remove(currV.point)  # blackedPixels += 1  # network_plot_3D(G, angle=0)
#                     elif nb_closest in X:
#                         # pervious_point = coor[ind[0][j]]
#                         # pervious_point = Vertex(pervious_point)
#                         # currV = Vertex(point)
#                         if not G.has_edge(str(point), str(arr[ind[0][j]])):
#                             currV = Vertex(point)
#                             pervious_point = Vertex(arr[ind[0][j]])
#                             G.add_node(str(currV.point), pos=currV.point)
#                             G.add_node(str(pervious_point.point), pos=pervious_point.point)
#                             G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV,
#                                                                                                 pervious_point))  # coor.remove(currV.point)  # network_plot_3D(G, angle=0)  # point_array = np.delete(point_array, i, axis=0)
#
#             if count_neigh == 0 and distances[0][1] < 15:
#                 disconect_tree = False
#                 # network_plot_3D(G, angle=0)
#
#                 for j in range(1, len(distances[0])):
#                     nb_closest = arr[ind[0][j]]
#                     if nb_closest not in X and distances[0][j] <=4:
#                         X.append(nb_closest)
#                         newV = Vertex(nb_closest)
#                         # blackedPixels += 1
#                         currV = Vertex(point)
#                         G.add_node(str(currV.point), pos=currV.point)
#                         G.add_node(str(newV.point), pos=newV.point)
#                         G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
#                         # blackedPixels += 1
#                         Find_neigh = True
#                         # coor.remove(currV.point)
#                         # network_plot_3D(G, angle=0)
#                         break
#
#                     elif nb_closest in X and distances[0][j] <=4:
#                         # currV = Vertex(point)
#                         # pervious_point = coor[ind[0][j]]
#                         # pervious_point = Vertex(pervious_point)
#                         if not G.has_edge(str(point), str(arr[ind[0][j]])):
#                             currV = Vertex(point)
#                             pervious_point = Vertex(arr[ind[0][j]])
#                             G.add_node(str(currV.point), pos=currV.point)
#                             G.add_node(str(pervious_point.point), pos=pervious_point.point)
#                             G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV,
#                                                                                                 pervious_point))  # coor.remove(currV.point)  # network_plot_3D(G, angle=0)
#
#                         break  # point_array = np.delete(point_array, i, axis=0)
#
#                 if Find_neigh == False:
#                     # for j in range(1, len(distances[0])):
#                     #     currV = Vertex(point)
#                     #     nb_closest = arr[ind[0][j]]
#                     #     nb_closest_Vertex = Vertex(nb_closest)
#                     #     if not G.has_edge(str(currV.point),
#                     #                       str(nb_closest_Vertex.point)) and nb_closest_Vertex.point > currV.point:
#                     #         X.append(nb_closest)
#                     #         newV = Vertex(nb_closest)
#                     #
#                     #         currV = Vertex(point)
#                     #         G.add_node(str(currV.point), pos=currV.point)
#                     #         G.add_node(str(newV.point), pos=newV.point)
#                     #         G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
#                     #
#                     #         # blackedPixels += 1
#                     #         # network_plot_3D(G, angle=0)
#                             i = i + 1
#                             blackedPixels += 1
#                             arr.remove(point)
#                             break
#             elif count_neigh == 0 and distances[0][1] > 15:  # remove the edge from ends points
#                 i = i + 1
#                 blackedPixels += 1
#                 arr.remove(point)
#                 break
#
#             # i = i + 1
#             # blackedPixels += 1
#
#             if len(arr) > 1:
#                 delpoint.append(currV.point)
#                 arr.remove(currV.point)  # remove the currv from the list and find the other neigbours
#             else:
#                 break
#
#             # queue = collections.deque()  # queue.append(point_array)  # queue.popleft()  # p = np.array(queue)
#
#         # X.popleft()
#
#         graphs.append(G)
#
#         # empty queue
#         # current graph is finished ->store it
#         # graphs.append(G)
#         G_composed = nx.compose_all(graphs)
#
#         # network_plot_3D(G,0)
#         # display(G)
#
#         # reset start
#         start = None  # H = nx.compose(H, G)
#
#     # end while
#     # mlab.show()
#
#     return G

import collections
def buildTree_all_branches(coor1, coor, trimmedGraph, endNodes_main_artery, start=None):
    # Convert input arrays to numpy arrays
    coor1 = np.array(coor1) # all nodes coordinate
    coor = np.array(coor) #coordinates trunk and main artery

    # Use set intersection to find matching coordinates
    mask = np.isin(coor1, coor, assume_unique=True)
    k = np.where(np.all(mask, axis=1))[0]

    # Use numpy indexing to remove matched coordinates
    arr = np.delete(coor1, k, axis=0)
    arr = arr.tolist()


    from ast import literal_eval
    for l in range(len(endNodes_main_artery)):
        endNodes_main_artery[l] = literal_eval(endNodes_main_artery[l])

    # Convert end nodes to numpy array and append to arr
    endNodes_main_artery = np.array(endNodes_main_artery)
    arr += endNodes_main_artery.tolist()

    # Sort arr by x-coordinate
    arr.sort(key=lambda x: x[0])

    # Initialize graph
    G = trimmedGraph.copy()

    # Set up KDTree
    point_array = np.array(arr)
    tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)

    queue = collections.deque()
    queue.append(endNodes_main_artery)
    X = []
    while len(queue[0]):
        point = queue[0][0]
        X.append(point)
        X= np.array(X)
        X = X.tolist()
        distances, ind = tree.query([point], k=6)
        count_neigh = 0
        for j in range(1, len(distances[0])):
            nb_closest = arr[ind[0][j]]

            if len(X)!=0 and nb_closest not in X and distances[0][j] <= 3:
                # dist = np.linalg.norm(nb_closest - point)
                # if dist <= 0.625:
                if not G.has_edge(str(point), str(nb_closest)):
                    X.append(nb_closest)
                    newV = Vertex(nb_closest)
                    currV = Vertex(point)
                    G.add_node(str(currV.point), pos=currV.point)
                    G.add_node(str(newV.point), pos=newV.point)
                    G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
                    count_neigh += 1
                    queue[0] = np.append(queue[0], [newV.point], axis=0)

            # elif len(X) != 0 and nb_closest in X and distances[0][j] <= 1:
            #     # dist = np.linalg.norm(nb_closest - point)
            #     # if dist <= 0.625:
            #     if not G.has_edge(str(point), str(nb_closest)):
            #         newV = Vertex(nb_closest)
            #         currV = Vertex(point)
            #         G.add_node(str(currV.point), pos=currV.point)
            #         G.add_node(str(newV.point), pos=newV.point)
            #         G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
            #         count_neigh += 1

        queue = queue[0]  # extract the numpy array from the tuple
        queue = np.delete(queue, 0, axis=0)  # remove the first row (axis=0)
        queue = ([queue])
        #queue.popleft()
        # network_plot_3D(G, 0)
    return G


    # # Initialize variables
    # X = []
    # blackedPixels = 0
    # nWhitePixels = len(arr)
    #
    # while nWhitePixels != blackedPixels:
    #
    #     # Find next unvisited point
    #     mask = np.logical_not(np.isin(point_array, X))
    #     if np.count_nonzero(mask) == 0:
    #         break
    #     point = point_array[np.argwhere(mask)[0][0]]
    #
    #     # Find neighbors within distance threshold
    #     distances, ind = tree.query([point], k=3)
    #     count_neigh = 0
    #     for j in range(1, len(distances[0])):
    #         nb_closest = coor[ind[0][j]]
    #         if nb_closest in X:
    #             # Add edge to existing node
    #             if not G.has_edge(str(point), str(nb_closest)):
    #                 currV = Vertex(point)
    #                 pervious_point = Vertex(nb_closest)
    #                 G.add_node(str(currV.point), pos=currV.point)
    #                 G.add_node(str(pervious_point.point), pos=pervious_point.point)
    #                 G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))
    #         else:
    #             # Add new node and edge
    #             dist = np.linalg.norm(nb_closest - point)
    #             if dist <= 0.625:
    #                 X.append(nb_closest)
    #                 newV = Vertex(nb_closest)
    #                 currV = Vertex(point)
    #                 G.add_node(str(currV.point), pos=currV.point)
    #                 G.add_node(str(newV.point), pos=newV.point)
    #                 G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
    #                 count_neigh += 1
    #                 blackedPixels += 1
    #             elif dist <= 4:
    #                 # Add new node and edge within larger distance threshold
    #                 n_inter_nodes = int(np.ceil(dist / 0.625)) - 1
    #                 delta = (nb_closest - point) / (n_inter_nodes + 1)
    #                 prev_node = point
    #                 for i in range(n_inter_nodes):
    #                     X.append(prev_node + delta)
    #                     newV = Vertex(prev_node + delta)
    #                     currV = Vertex(prev_node)
    #                     G.add_node(str(currV.point), pos=currV.point)
    #                     G.add_node(str(newV.point), pos=newV.point)
    #                     G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))
    #                     prev_node = prev_node + delta
    #                     count_neigh += 1
    #                     blackedPixels += 1
    #                 X.append
    #
    #         return G




def mergeEdges(graph):

    # copy the graph
    g = graph.copy()

    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

    nb = []


    for v0 in startNodes:

        if v0 not in g:
            startNNbs = v0
            continue

        # start a line traversal from each neighbor
        startNNbs = list(nx.neighbors(g, v0))




        if not len(list(startNNbs)): #Joyce added list bcos of "dict-keyiterator" has no len() error
            continue

        counter = 0
        v1 = startNNbs[counter]  # next nb of v0
        while True:

            if nx.degree(g, v1) == 2:
                # we have a node which has 2 edges = this is a line segement
                # make new edge from the two neighbors
                nbs = list(nx.neighbors(g, v1))

                # if the first neihbor is not n, make it so!

                if nbs[0] != v0:
                    nbs.reverse()

                pxL0 = g[v0][v1]["object"].pixels  # the pixel list of the edge 0
                pxL1 = g[v1][nbs[1]]["object"].pixels  # the pixel list of the edge 1
                if len(pxL1) > 0:
                    print("yes")

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0, nbs[1],
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1] + pxL1))

                # delete the node n
                nb.append(v1)

                g.remove_node(v1)

                # set v1 to new left node
                v1 = nbs[1]

            else:
                counter += 1
                if counter == len(startNNbs):
                    break
                v1 = startNNbs[counter]  # next nb of v0
    print(nb)
    # weight the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)
        if len(o.pixels) != 0:
            values = o.pixels
            record = {}
            for i in range(len(values)):
                record[radii[values[i]]] = values[i]
            keys_list = list(record.keys())
            if len(keys_list) != 0:
                average_radii = sum(keys_list) / len(keys_list)
            else:
                average_radii = 0

            g[u][v]["radius"] = average_radii

        # else:


    # network_plot_3D(g, 0)
    # display(g)
    return g


def remove_small_seprate_branches(graph):
    g = graph.copy()
    endNodes = getEndNodes(g)
    edgesToRemove = []

    seprateNodes = [seprateN for seprateN in g.nodes() if nx.degree(g, seprateN) == 0]
    for n in seprateNodes:
        g.remove_node(n)


    for u, v, o in g.edges(data="object"):
        if u in endNodes and v in endNodes:
            # if o is not None and hasattr(o, 'pixels') and len(o.pixels) < 10:
                edgesToRemove.append((u, v))

    for u, v in edgesToRemove:
        # if g[u][v]["weight"] < 5:
        g.remove_edge(u, v)
        g.remove_node(u)
        g.remove_node(v)
        # else:
        #     print("This is seprate tree with weight > 5")

    return g, edgesToRemove

def vessel_radius_from_sigma(self,scale):
    #return self.spacing*math.sqrt(2)*sigma
    mask = scale< (2./np.sqrt(2)*self._sigma0)
    rad=np.zeros(mask.shape)
    rad[mask]=np.sqrt(2.)*(np.sqrt((scale[mask]*self._spacing)**2.0 + (self._sigma0*self._spacing)**2.0) -0.5*self._sigma0*self._spacing)
    rad[~mask]=np.sqrt(2.)*(np.sqrt((scale[~mask]*self._spacing)**2.0 + (self._sigmap*self._spacing)**2.0) -0.5*self._sigmap*self._spacing)
    return rad

import ast
import math
def coneccted_tree(graph, radii):
    # copy the graph
    g = graph.copy()

    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) == 1]
    middleNodes = [startN for startN in g.nodes() if nx.degree(g, startN) == 2]

    nb = []
    x = []
    points = list(g.nodes())
    points = [ast.literal_eval(string) for string in points]
    edges = list(g.edges())
    for v0 in startNodes:
        # g = g.copy()

        # # start a line traversal from each neighbor
        # startNNbs = list(nx.neighbors(g, v0))
        #
        #
        #
        # counter = 0
        # v1 = startNNbs[counter]  # next nb of v0
        # lists = [ast.literal_eval(string) for string in startNodes] # convert list of stings to list of list, and then NumpyArray
        # point_array = np.array(lists)

        point_array = np.array(points)
        point = ast.literal_eval(v0) # convert 'v0' to v0
        point = np.array(point)


        tree = KDTree(point_array, leaf_size=point_array.shape[0] + 1)
        # tree = KDTree(b, leaf_size=b.shape[0] + 1)

        # point_array = list(coor)
        point_array = list(g.nodes())
        point_array = [ast.literal_eval(string) for string in point_array]

        count_neigh = 0

        if len(point_array) > 2:
            k = [k for k in range(2, 5)]
            distances, ind = tree.query([point], k=5)


        elif len(point_array) == 2:
            distances, ind = tree.query([point], k=2)


        else:
            distances, ind = tree.query([point], k=1)
            break

        for j in range(1, len(distances[0])):
            # nb_closest = coor[ind[0][j]]
            nb_closest = points[ind[0][j]]

            if distances[0][j] <= 10:
                if not g.has_edge(str(list(point)), str(nb_closest)) and not nx.has_path(g, str(list(point)), str(nb_closest)):  # 1.5033
                    # index_point = points.index(str(list(point)))
                    # edge = edges[index_point]
                    edges_of_node = []
                    for edge in edges:
                        edge = list(edge)
                        edge = [eval(a) for a in edge]
                        if list(point) in edge:
                            edges_of_node.append(edge)
                            break

                    print(edges_of_node)
                    for n in range(len(edges_of_node[0])):
                        if edges_of_node[0][n] != list(point):
                            next_point = edges_of_node[0][n]

                    p1 = list(point)
                    p2 = next_point
                    p3 = nb_closest
                    direction1 = np.array(p2) - np.array(p1)
                    direction2 = np.array(p3) - np.array(p1)
                    norm_direction1 = direction1 / np.linalg.norm(direction1)
                    norm_direction2 = direction2 / np.linalg.norm(direction2)

                    cosine_angle = np.dot(norm_direction1, norm_direction2)
                    degree_angle = abs(math.degrees(cosine_angle))
                    angle = 50
                    if degree_angle >= angle:

                        if nb_closest not in x:
                            currV = Vertex(point)
                            new_point = Vertex(nb_closest)
                            #g.add_node(str(currV.point), pos=currV.point)
                            #g.add_node(str(new_point.point), pos=new_point.point)
                            g.add_edge(str(list(currV.point)), str(new_point.point), object=Edge(currV, new_point))
                            x.append(list(new_point.point))
                            x.append(list(currV.point))
                            #g.remove_node(str(currV.point))
                            new_path = currV.point+new_point.point
                            #new_path = list(new_path)
                            # node_with_commas = new_path.replace(' ', ', ')
                            # print(node_with_commas)
                            # result = '[' + node_with_commas.strip(', ') + ']'
                            # print(result)
                            # g.remove_node(str(new_path))
                            #nx.add_path(g, new_path)
                            #filtered_nodes = [node for node in g.nodes() if
                            #                 not isinstance(node, float) or node not in new_path]
                            #g.add_nodes_from(filtered_nodes)


                            break
                    else:
                        break

    return g




def getEndNodes(g):
    """Returns end nodes from graph G with degree == 1 """
    return [n for n in nx.nodes(g) if nx.degree(g, n) == 1]
    # return [n for n in nx.nodes_iter(g) if nx.degree(g, n) == 1]

def removeSmallEdge(graphmerged):
    """Removes small edges from the graph which are not vessel like. Short edges appear due to voronoi
    partition. These add up false edges. Small edges are removed using euclidean distance and weigh of
    each edge"""
    g = graphmerged.copy()

    #get all ending nodes
    endNodes = getEndNodes(g)

    for p in endNodes:
        endNNbs = list(nx.neighbors(g, p))
        endChosen = [x for x in g.nodes() if x in endNodes and x in endNNbs]
        for q in endChosen:
            if p or q is str:
                from ast import literal_eval
                p = literal_eval(p)
                q = literal_eval(q)
                p = np.array(p)
                q = np.array(q)
                dist = np.linalg.norm(p - q)
            else:
                dist = np.linalg.norm(np.array(p.point) - np.array(q.point))
                # print(dist,"euclediandistance",g[p][q]["weight"],"weight of Branch!!!!!!!!!!!!!!!!")
                if g[p][q]["weight"] > dist and dist < 1:
                    g.remove_node(p)
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                elif g[p][q]["weight"] < 1:
                    g.remove_node(p)
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # g.remove_node(q)


    # start with nodes having 3 or more neighbors
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) >= 3]
    # print("start nodes",startNodes)
    # i = 0
    for v0 in startNodes:
        # print("v0",v0)
        # start a line traversal from each neighbor
        startNNbs = list(nx.neighbors(g, v0))
        # print("neighbors",startNNbs)
        #for a node both neighbor and ending
        chosen = [x for x in g.nodes() if x in endNodes and x in startNNbs]
        for x0 in chosen:
            # if i == 0:
            g.add_node(x0)
            # else:
            if g[v0][x0]["weight"] < 1:  # think honiiiiiiiiiiiiii
                    # print("chosen weight",g[v0][x0]["weight"])
                g.remove_node(x0)
            # i=i+1

    return g

def pointPicking (pixelList,weight):
    """returns list of pixels between 10-90% of whole list"""
    newList = pixelList[int(len(pixelList) * .0): int(len(pixelList) * .99)]
    return newList

def pixelDistance(pixelList):
    """From list of pixels, calculates euclidean distances between each pixels and returns total distance"""
    distance = 0
    a = pixelList
    # if len(pixelList) >= 20:
    #     a = pixelList
    # else:
    #     a = pixelList[::2]
    #     # a = pixelList
    # print("every second item",a)
    allpairs = []
    allpairs.append(a[0])
    for i in range(len(a[1])):
        allpairs.append(a[1][i])
    allpairs.append(a[-1])
    # pairs = [(a[i],a[i+1]) for i in range(len(a)-1)]
    pairs = [(allpairs[i],allpairs[i+1]) for i in range(len(allpairs)-1)]
    for p, q in pairs:
        p = pd.eval(p)
        q = pd.eval(q)
        distance = distance + np.linalg.norm(np.array(p) - np.array(q))
        print(np.linalg.norm(np.array(p) - np.array(q)))
    print(a)
    print(pairs)
    print("distance",distance,"weight",len(a))
    return distance

def radialDistance(middlePixels,radii): #new_img_dst
    """Uses middle pixels and distance transform of vessel mask to return distance of centerline
    pixel from the nearest background, thereby calculating the radial distance of each vessel segment.
    Returns cross-sectional area and average radius from top 20% of vessel radius"""
    radiii = []
    for i in range(0,len(middlePixels)):
        R = middlePixels[i]
        # R = pd.eval(R)
        # key_list = list(radii.keys())
        # val_list = list(radii.values())
        # position = val_list.index(R)
        # radiii.append(key_list[position])
        # print(key_list[position])
        radiii.append(radii[R])




    if len(radiii) != 0:
        radiii.sort(reverse=True)
        # toptrim = stats.trim1(radii,proportiontocut=0.1,tail='right')
        trimmedradius = stats.trim1(radiii,proportiontocut=0.9,tail='left')
        avg_radius = statistics.mean(trimmedradius)
        diameter = avg_radius * 2
    else:
        avg_radius = 0
        diameter = 0
    crossSecArea = pi * avg_radius ** 2
    return crossSecArea,avg_radius, diameter

def avgBranchRadius (combinedGraph,radii):  #new_img_dst
    """Input: Built graph, distance transform of vesselmask
    First calculates additional distances of ending nodes from its background and adds it to the
    branch length. Next, for longer branches with distance > 5, pick intermediate pixels and calculate
    radial distance. Add length, radius, cross-sectional area to graph dataset. Repeat for smaller edges <= 5,
    only difference is that  intermediate pixels are only 3 pixels of the middle ones."""

    # get all ending nodes
    for u, v, o in combinedGraph.edges(data="object"):
        combinedGraph[u][v]["weight"] = len(o.pixels)
        print("#######branch radius estimation#########")

        combinedGraph[u][v]["type"] = "branchType"


        allpoints = []
        weight = combinedGraph[u][v]["weight"]
        allpoints.append(u)
        allpoints.append(o.pixels)
        allpoints.append(v)
        pixelDist = pixelDistance(allpoints)
        # pixelDist = (pixelDist / weight) * weight

        # print o.pixels



        if weight >= 5:
            betweenPxlList = pointPicking(o.pixels,weight)
            csaAvg, radiusAvg, diameter = radialDistance(betweenPxlList,radii)  #new_img_dst
            ##################################################################
            # if weight < dist:
            #     print("weight", weight, "less than distance", dist)
            #     weight = dist

                # weight = (dist + weight) / 2
            # print("longer branch","CSA",csaAvg,"weight",weight+endNodeAddition,"distance",pixelDist+endNodeAddition,
            #       "volume",csaAvg*(weight+endNodeAddition))
            # csaAvg = (csaMax + csaAvg) / 2
            # csaAvg = csaMax
            # volume = csaAvg * (pixelDist)
            combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition In my small vascular tree, I do not need endnodeadddition since the center line in my example start from center of first circle
            combinedGraph[u][v]["radius"] = radiusAvg
            combinedGraph[u][v]["diameter"] = diameter
            combinedGraph[u][v]["RatioLendiameter"] = pixelDist / diameter
            combinedGraph[u][v]["csa"] = csaAvg
            # combinedGraph[u][v]["volume"] = volume

            ###############################################################
            # avgVolume = segmentedBranchVolume(pixelList=o.pixels,new_img_dst=new_img_dst,endNodeAddition=endNodeAddition)
            # combinedGraph[u][v]["csa"] = csaAvg
            # combinedGraph[u][v]["volume"] = avgVolume
            ###################################################################

        elif weight < 5 and weight >= 1:
            mid = weight//2
            csaSmall,radiusSmall, diameter = radialDistance(o.pixels[mid-1:mid+1],radii)  #new_img_dst
            # volumeS = csaSmall * (pixelDist + endNodeAddition)
            combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition
            combinedGraph[u][v]["radius"] = radiusSmall
            combinedGraph[u][v]["diameter"] = diameter
            combinedGraph[u][v]["RatioLendiameter"] = pixelDist / diameter
            combinedGraph[u][v]["csa"] = csaSmall
            # combinedGraph[u][v]["volume"] = volumeS
            # print("smaller branch","CSA", csaSmall,"weight",weight,"volume",csaSmall*(weight + endNodeAddition))
        # elif  weight == 0:


        else:

            combinedGraph[u][v]["length"] = 0
            combinedGraph[u][v]["radius"] = 0
            combinedGraph[u][v]["diameter"] = 0
            combinedGraph[u][v]["RatioLendiameter"] = 0
            combinedGraph[u][v]["csa"] = 0
            # combinedGraph[u][v]["volume"] = 0
            print("0 branch weight", weight)

            continue

    return combinedGraph

def avgBranchRadius_group2(combinedGraph,radii):  #new_img_dst
    """Input: Built graph, distance transform of vesselmask
    First calculates additional distances of ending nodes from its background and adds it to the
    branch length. Next, for longer branches with distance > 5, pick intermediate pixels and calculate
    radial distance. Add length, radius, cross-sectional area to graph dataset. Repeat for smaller edges <= 5,
    only difference is that  intermediate pixels are only 3 pixels of the middle ones."""

    # get all ending nodes
    for u, v, o in combinedGraph.edges(data="object"):

            # combinedGraph[u][v]["weight"] = len(o.pixels)
            print("#######branch radius estimation#########")

            combinedGraph[u][v]["type"] = "branchType"

            weight = combinedGraph[u][v]["weight"]
            allpoints = []
            allpoints.append(u)
            allpoints.append(o.pixels)
            allpoints.append(v)
            pixelDist = pixelDistance(allpoints)
            # pixelDist = (pixelDist / weight) * weight

            # print o.pixels
            if len(o.pixels)!=0:
                combinedGraph[u][v]["weight"] = len(o.pixels)
                weight = combinedGraph[u][v]["weight"]

                if weight >= 5:
                    betweenPxlList = pointPicking(o.pixels,weight)
                    csaAvg, radiusAvg, diameter = radialDistance(betweenPxlList,radii)  #new_img_dst
                    ##################################################################
                    # if weight < dist:
                    #     print("weight", weight, "less than distance", dist)
                    #     weight = dist

                        # weight = (dist + weight) / 2
                    # print("longer branch","CSA",csaAvg,"weight",weight+endNodeAddition,"distance",pixelDist+endNodeAddition,
                    #       "volume",csaAvg*(weight+endNodeAddition))
                    # csaAvg = (csaMax + csaAvg) / 2
                    # csaAvg = csaMax
                    # volume = csaAvg * (pixelDist)
                    combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition In my small vascular tree, I do not need endnodeadddition since the center line in my example start from center of first circle
                    combinedGraph[u][v]["radius"] = radiusAvg
                    combinedGraph[u][v]["diameter"] = diameter
                    combinedGraph[u][v]["RatioLendiameter"] = pixelDist / diameter
                    combinedGraph[u][v]["csa"] = csaAvg
                    # combinedGraph[u][v]["volume"] = volume

                    ###############################################################
                    # avgVolume = segmentedBranchVolume(pixelList=o.pixels,new_img_dst=new_img_dst,endNodeAddition=endNodeAddition)
                    # combinedGraph[u][v]["csa"] = csaAvg
                    # combinedGraph[u][v]["volume"] = avgVolume
                    ###################################################################

                elif weight < 5 and weight >= 1:
                    mid = weight//2
                    csaSmall,radiusSmall, diameter = radialDistance(o.pixels[mid-1:mid+1],radii)  #new_img_dst
                    # volumeS = csaSmall * (pixelDist + endNodeAddition)
                    combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition
                    combinedGraph[u][v]["radius"] = radiusSmall
                    combinedGraph[u][v]["diameter"] = diameter
                    combinedGraph[u][v]["RatioLendiameter"] = pixelDist / diameter
                    combinedGraph[u][v]["csa"] = csaSmall
                    # combinedGraph[u][v]["volume"] = volumeS
                    # print("smaller branch","CSA", csaSmall,"weight",weight,"volume",csaSmall*(weight + endNodeAddition))
                # elif  weight == 0:

            else:

                if combinedGraph[u][v]['weight'] == 0:
                    combinedGraph[u][v]["length"] = 0
                    combinedGraph[u][v]["radius"] = 0
                    combinedGraph[u][v]["diameter"] = 0
                    combinedGraph[u][v]["RatioLendiameter"] = 0
                    combinedGraph[u][v]["csa"] = 0
                    # combinedGraph[u][v]["volume"] = 0
                    print("0 branch weight", weight)

                    continue
                elif combinedGraph[u][v]['weight'] == -1:
                    continue

    return combinedGraph

def connected_component_to_graph(coor,wholeGraph):
    """Input: Centerline
    Takes each connected component from centerline, buildTree creates graph, mergeEdges merges
    neighbouring pixels and forms longer edges joining between pixels, removeSmallEdge removes
    false edges, joins all individual graphs using union function. Finally removes isolated nodes"""
    s = scipy.ndimage.generate_binary_structure(3, 3)
    coor = np.array(coor)
    coor = coor.reshape((coor.shape[0], coor.shape[1], 1))
    label_im, nb_labels = scipy.ndimage.label(coor,structure=cube(3))
    unique, counts, indices = np.unique(label_im, return_counts=True, return_index=True)
    # print (unique, counts, indices)


    for each_label in range(0,nb_labels+1):
        if counts[each_label] != 0:
            print ("Label ID of component",each_label)
            each_component = np.where(label_im == unique[each_label], label_im, 0)
            index = np.where(label_im == each_label)
            startAt = [x[0] for x in index]
            print ("Starting point of Graph",startAt)
            componentGraph, G = buildTree(each_component, start=startAt)
            mergedGraph = mergeEdges(G)
            # mergedGraph = G #graph with Allnodes between start and end nodes for each branch

            trimmedGraph = removeSmallEdge(mergedGraph)
            # remergedGraph = mergeEdges(trimmedGraph) #

            wholeGraph = nx.union(wholeGraph , mergedGraph)  #
    wholeGraph.remove_nodes_from(list(nx.isolates(wholeGraph))) #remove isolated nodes without neighbouring nodes
    return wholeGraph


# def network_plot_3D(G, angle,linethickness=1, save=False,csa = False):
#     # Get node positions
#     pos = nx.get_node_attributes(G, 'pos')
#
#     # Get number of nodes
#     n = G.number_of_nodes()
#     print ("number"+str(n))
#
#     with plt.style.context(('ggplot')):
#
#         fig = plt.figure(figsize=(10, 7))
#         ax = Axes3D(fig)
#
#         # Loop on the pos dictionary to extract the x,y,z coordinates of each node
#         for key, value in pos.items():
#             xi = value[0]
#             yi = value[1]
#             zi = value[2]
#
#
#
#
#             # Scatter plot
#             ax.scatter(xi, yi, zi,  s=20, alpha=0.7,c='blue') #c=colors[key]
#
#
#         # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
#         # Those two points are the extrema of the line to be plotted
#         for i, j in enumerate(G.edges()):
#
#             x = np.array((pos[j[0]][0], pos[j[1]][0]))
#             y = np.array((pos[j[0]][1], pos[j[1]][1]))
#             z = np.array((pos[j[0]][2], pos[j[1]][2]))
#
#             if csa == False:
#                 # Plot the connecting lines
#                 ax.plot(x, y, z, c='black', alpha=0.5, linewidth=linethickness)
#
#     # Set the initial view
#     ax.view_init(30, angle)
#
#     plt.show()
#
#
#     return


def network_plot_3D(G, angle,linethickness=1, save=False,csa = False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')


    for v in G.nodes():
        if v not in pos:
            pos[v] = tuple(ast.literal_eval(v))

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w")

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _format_axes(ax)
    fig.tight_layout()
    plt.show()

    return




def writeGraph(G):
    """Write edges of graph in xml format"""


    nx.write_edgelist(G, '/home/arah607/Desktop/outputGraph/Graph_15814w_g1g2.csv', delimiter=',' ,  data=['radius', 'diameter', 'length','RatioLendiameter', 'ratioDiam', 'angle']) # #tree_CIP_right_13febw # test_smallpart_15814w_withoutsidebranch.csv




def Connceted_Tree(endNodes_main_artery, data_file, file_name):
    from csv import writer

    with open(file_name, 'w', newline='') as write_obj:
        df11 = pd.read_csv(data_file, header=None)
        df11 = df11.to_numpy()
        endNodes_main_artery = np.array(endNodes_main_artery)
        csv_writer = writer(write_obj)
        count = 0
        choise_row = []

        # for row in df11:
        #     for row1 in range(len(endNodes_main_artery)):
        #         find_idx = np.where(np.all(df11[:, 3:6] == endNodes_main_artery[3:6], axis=1))




        for row in df11:
            row = [float(i) for i in row]

            if row[0:3] != row[3:6] and row not in choise_row:
                child_indx = np.where(np.all(df11[:, 0:3] == row[3:6], axis=1))
                child_indx = np.asarray(child_indx)

                if child_indx.size!=0:
                    # choise_row.append(row)
                    csv_writer.writerow(row)
                    for j in range(len(child_indx)):
                         child_indx = list(child_indx)
                         choise_row.append(list(df11[child_indx[0][j]]))
                         csv_writer.writerow(df11[child_indx[0][j]])


                # csv_writer.writerow(choise_row)
                count = count + 1

            else:
                count = count + 1


# if row == row0:
#     child_indx = np.where(np.all(df11[:, 0:3] == row0[3:6], axis=1))
#     child_indx = np.asarray(child_indx)
#
#     if child_indx.size != 0:
#         # choise_row.append(row)
#         csv_writer.writerow(row)
#         for j in range(len(child_indx)):
#             child_indx = list(child_indx)
#             choise_row.append(list(df11[child_indx[0][j]]))
#             csv_writer.writerow(df11[child_indx[0][j]])
#         for k in range(len(child_indx)):
#             row = df11[k]
#             if row[0:3] != row[3:6] and row not in choise_row:
#                 child_indx1 = np.where(np.all(df11[:, 0:3] == row[3:6], axis=1))
#                 child_indx1 = np.asarray(child_indx1)
#                 for n in range(len(child_indx1)):
#                     child_indx1 = list(child_indx1)
#                     choise_row.append(list(df11[child_indx1[0][n]]))
#                     csv_writer.writerow(df11[child_indx1[0][n]])

def euclidean_distance(node1, node2):
    node1 = np.array(eval(node1))  # Convert string to numpy array
    node2 = np.array(eval(node2))  # Convert string to numpy array
    return np.linalg.norm(node2 - node1)

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norms)
    return np.degrees(angle)

from collections import defaultdict

def All_subgraphs(graph):
    g = graph.copy()

    # components = nx.connected_components(g)
    components = list(nx.weakly_connected_components(g))

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    subgraphs = []

    main_graph = g

    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]



        subgraph= nx.DiGraph()
        subgraph.add_nodes_from(nodes_in_label)
        subgraph.add_edges_from(edges_in_label)

        # node_positions={}
        # for i in range(len(nodes_in_label)):
        #     node_positions[nodes_in_label[i]] =  tuple(ast.literal_eval(nodes_in_label[i]))
        #
        # nx.set_node_attributes(subgraph, node_positions, 'pos')

        # Copy attributes from the original graph to the subgraph
        for node in nodes_in_label:
            subgraph.nodes[node].update(g.nodes[node])

        # Copy edge attributes from the original graph to the subgraph
        for u, v in edges_in_label:
            subgraph[u][v].update(g[u][v])

        subgraphs.append(subgraph)

        # main_graph = nx.compose(g, subgraph)

    return subgraphs



def get_directed_subgraphs(main_graph, root_branches):
    subgraphs = []

    # Iterate through all nodes in the graph and assign positions programmatically
    for node in main_graph.nodes():
       if 'pos' not in main_graph.nodes[node]:
            # Set the 'pos' attribute for the node
            main_graph.nodes[node]['pos'] = tuple(ast.literal_eval(node))

    # Ensure root branches are still present in the modified graph
    valid_root_branches = [root for root in root_branches if root in main_graph.nodes]

    for root in valid_root_branches:
        # Create a directed subgraph starting from the root node
        subgraph = nx.DiGraph()
        visited_nodes = set()
        stack = [root]

        while stack:
            current_node = stack.pop()
            if current_node not in visited_nodes:
                visited_nodes.add(current_node)
                subgraph.add_node(current_node)

                # Add successors to the stack for traversal
                stack.extend(main_graph.successors(current_node))

                # Add edges to the subgraph
                for successor in main_graph.successors(current_node):
                    subgraph.add_edge(current_node, successor)

                # Copy 'pos' attribute if present in the original graph
                if 'pos' in main_graph.nodes[current_node]:
                    subgraph.nodes[current_node]['pos'] = main_graph.nodes[current_node]['pos']

        subgraphs.append(subgraph)

    return subgraphs

from scipy.spatial.distance import euclidean
def calculate_distance(graph1, graph2):
    terminal_nodes1 = [node for node, degree in graph1.degree() if degree == 1]
    terminal_nodes2 = [node for node, degree in graph2.degree() if degree == 1]

    min_distance_thresholder = 10  #float('inf')
    min_distance = None
    closest_nodes = (None, None)
    found_valid_distance = False


    for node1 in terminal_nodes1:
        for node2 in terminal_nodes2:
            if 'pos' in graph1.nodes[node1] and 'pos' in graph2.nodes[node2]:
                pos1 = np.array(nx.get_node_attributes(graph1, 'pos')[node1])
                pos2 = np.array(nx.get_node_attributes(graph2, 'pos')[node2])
                # if pos1 is None or pos2 is None:
                #     continue  # Skip nodes with missing positions

                distance = euclidean(pos1, pos2)
                # min_distance = min(min_distance, distance)
            else:
                pos1 = None  # Handle missing 'pos' attribute
                pos2 = None  # Handle missing 'pos' attribute

            if pos1 is not None and pos2 is not None:
                distance = euclidean(pos1, pos2)

                if distance < min_distance_thresholder:

                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        closest_nodes = (node1, node2)
                        found_valid_distance = True

    if not found_valid_distance:
        min_distance = None
        closest_nodes = (None, None)


    return min_distance, closest_nodes



def connect_separate_GraphTrees(subgraphs, graph, radi):

    g = graph.copy()

    distances = {}
    # main_graph = g
    graphs = subgraphs

    # for subgraph in graphs:
    #     main_graph = nx.compose(main_graph, subgraph)

    connected_graphs = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            graph1 = graphs[i]
            graph2 = graphs[j]
            if graph1 not in connected_graphs or graph2 not in connected_graphs:
                distance, closestNodes = calculate_distance(graph1, graph2)
                print("Distance:", distance)
                # if distance != None:
                if distance is not None and distance <5:
                    print("Adding edge between", closestNodes[0], "and", closestNodes[1])
                    g.add_edge(closestNodes[0], closestNodes[1])
                    def str_to_tuple(s):
                        # Remove the square brackets and split by commas
                        parts = s.strip('[]').split(',')
                        # Convert parts to floats and create a tuple
                        return tuple(float(x.strip()) for x in parts)

                    # Convert the strings to numeric tuples
                    point_a = str_to_tuple(closestNodes[0])
                    point_b = str_to_tuple(closestNodes[1])

                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
                    g[closestNodes[0]][closestNodes[1]]["length"] = distance #+ endNodeAddition
                    g[closestNodes[0]][closestNodes[1]]["radius"] = max(radi[closestNodes[0]], radi[closestNodes[1]])
                    g[closestNodes[0]][closestNodes[1]]["diameter"] = 2*max(radi[closestNodes[0]], radi[closestNodes[1]])
                    g[closestNodes[0]][closestNodes[1]]["RatioLendiameter"] = distance / (2*max(radi[closestNodes[0]], radi[closestNodes[1]]))
                    # main_graph[closestNodes[0]][closestNodes[1]]["csa"] = csaSmall


                    distances[(i, j)] = distance
                    connected_graphs.append(graph1)
                    connected_graphs.append(graph2)

    return  g





def join_separate_GraphTrees(subgraphs, graph, radi):

    g = graph.copy()

    distances = {}
    # main_graph = g
    graphs = subgraphs

    # for subgraph in graphs:
    #     main_graph = nx.compose(main_graph, subgraph)

    connected_graphs = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            graph1 = graphs[i]
            graph2 = graphs[j]
            if graph1 not in connected_graphs or graph2 not in connected_graphs:
                distance, closestNodes = calculate_distance(graph1, graph2)
                print("Distance:", distance)
                # if distance != None:
                if distance is not None:
                    print("Adding edge between", closestNodes[0], "and", closestNodes[1])
                    g.add_edge(closestNodes[0], closestNodes[1])
                    def str_to_tuple(s):
                        # Remove the square brackets and split by commas
                        parts = s.strip('[]').split(',')
                        # Convert parts to floats and create a tuple
                        return tuple(float(x.strip()) for x in parts)

                    # Convert the strings to numeric tuples
                    point_a = str_to_tuple(closestNodes[0])
                    point_b = str_to_tuple(closestNodes[1])

                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
                    g[closestNodes[0]][closestNodes[1]]["length"] = distance #+ endNodeAddition
                    g[closestNodes[0]][closestNodes[1]]["radius"] = max(radi[closestNodes[0]], radi[closestNodes[1]])
                    g[closestNodes[0]][closestNodes[1]]["diameter"] = 2*max(radi[closestNodes[0]], radi[closestNodes[1]])
                    # g[closestNodes[0]][closestNodes[1]]["RatioLendiameter"] = distance / (2*max(radi[closestNodes[0]], radi[closestNodes[1]]))
                    # main_graph[closestNodes[0]][closestNodes[1]]["csa"] = csaSmall


                    distances[(i, j)] = distance
                    connected_graphs.append(graph1)
                    connected_graphs.append(graph2)

    return  g



# def calculate_distance(node1, node2):
#     # Implement your distance calculation method here.
#     # For example, if nodes are represented as (x, y) tuples:
#     distance = np.linalg.norm(np.array(node1) - np.array(node2))
#     # Return the distance.
#     return distance
# def find_closest_nodes(graph1, graph2):
#     min_distance = float('inf')
#     closest_nodes = None
#
#     for node1 in graph1.nodes:
#         for node2 in graph2.nodes:
#             distance = calculate_distance(node1, node2)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_nodes = (node1, node2)
#
#     return closest_nodes, min_distance
#
#
# def join_subgraphs_by_distance(subgraphs):
#     main_graph = nx.Graph()
#
#     for i, subgraph1 in enumerate(subgraphs):
#         for j, subgraph2 in enumerate(subgraphs):
#             if i != j:  # Avoid comparing a subgraph with itself
#                 closest_nodes, min_distance = find_closest_nodes(subgraph1, subgraph2)
#                 if closest_nodes is not None:
#                     main_graph.add_edge(*closest_nodes, length=min_distance)
#
#     return main_graph



def Final_graph(graph, rad):
    g = graph.copy()

    components = nx.connected_components(g)

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    label_root_store = {}
    separate_directed_graphs = nx.DiGraph()
    save_RootBranch = []
    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]

        points = []
        radiiii = []
        lengths = []
        label_branches = defaultdict(list)




        for u, v in edges_in_label:

            points.append(u)
            points.append(v)
            if 'diameter' not in g[u][v]:
                print("######################this branch doesnt have diameter######################")
                # g[u][v]['diameter'] = 0.8
                diameter = 0.8
                # g[u][v]['length'] = 1.5
                length = 1.5
                radiiii.append(diameter)
                lengths.append(length)
                label_branches[label_array[i]].append((diameter, length))
            else:
                diameter = g[u][v]['diameter']
                length = g[u][v]['length']
                points.append(diameter)
                radiiii.append(diameter)
                lengths.append(length)
                label_branches[label_array[i]].append((diameter, length))

        radiiii.sort(reverse = True)
        biggest_branches = radiiii[:2]
        # bigStartBranch = max(radiiii)
        # bigStartLength = max(lengths)

        targetNode = []
        test = []
        # for u, v, attr in g.edges(data=True):
        #   diameter_difference = abs(biggest_branches[0] - biggest_branches[1])
        #   if diameter_difference < 1.5:
        #       choosLen = []
        #       for diameter, length in label_branches[1]:
        #           if diameter in biggest_branches:
        #               choosLen.append(length)
        #
        #       a = max(choosLen)
        #       Choosdiameter = next(diameter for diameter, length in label_branches[1] if diameter in biggest_branches and length == a)
        #       attr['diameter'] = Choosdiameter
        #   # diameter = attr['diameter']
        #   # if diameter in biggest_branches:
        #   #       test.append(diameter)
        #   # a = abs(test[0] - test[1])
        #   # print(test)
        #   # if a < 1.5 :
        #   #       longest_branch = max(label_branches[label_array[i]], key=lambda x: x[1])
        #   #       attr['diameter'] = longest_branch[0]
        #
        #   else:
        #         attr['diameter'] = max(biggest_branches[0], biggest_branches[1])
        #     # if attr['diameter'] == bigStartBranch:

        # g1 = nx.Graph(edges_in_label)
        # all_paths = nx.algorithms.simple_paths.all_simple_paths(g1, source=edges_in_label[0][0], target=edges_in_label[-1][-1])
        # longest_path = max(all_paths, key=len)

        # dag = nx.DiGraph()
        dag = nx.Graph()
        dag.add_nodes_from(nodes_in_label)
        dag.add_edges_from(edges_in_label)
        for source, target in dag.edges():
            # Assuming edge attributes are stored as dictionaries in the main graph
            edge_attributes = g[source][target]

            # Transfer attributes to the separate graph's edge
            dag[source][target].update(edge_attributes)


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

        # if path not in all_paths:
        #     nodes = dag.nodes()
        #     choosN = []
        #     for N in nodes:
        #         if nx.degree(dag, N) == 1:
        #             choosN.append(N)
        #
        #     all_paths = set()
        #     for node1 in choosN:
        #         for node2 in choosN:
        #             if node1 != node2:
        #                 paths = nx.all_simple_edge_paths(dag, source=node1, target=node2)
        #                 for path in paths:
        #                     all_paths.add(tuple(path))


        path_distances = []
        if len(all_paths) != 0:
            for path in all_paths:
                distance = 0
                for i in range(len(path) - 1):
                    distance += euclidean_distance(path[i], path[i + 1])
                path_distances.append((path, distance))

            # Sort the paths by distance in descending order
            path_distances.sort(key=lambda x: x[1], reverse=True)

            # Print the longest path
            longest_path, longest_distance = path_distances[0]
            print("Longest Path:", longest_path)
            print("Distance:", longest_distance)
            c =[]
        else:
            print("################ there is no path############################")
            g.remove_nodes_from(dag.nodes())
            g.remove_edges_from(dag.edges())

            continue
        check_edge_radius = {}
        if len(edges_in_label) <= 3:
            for p in range(len(edges_in_label)):
                u = edges_in_label[p][0]
                v = edges_in_label[p][1]
                check_edge_radius[edges_in_label[p]] = g[u][v]['radius']
            all_edges_below_threshold = all(radius < 3 for radius in check_edge_radius.values())
            if all_edges_below_threshold:
                g.remove_nodes_from(dag.nodes())
                g.remove_edges_from(dag.edges())
                continue


        # for i in range(len(longest_path)):
        #   if nx.degree(g, longest_path[i]) == 1:
        #       c.append(longest_path[i])
        if  nx.degree(g, longest_path[0]) == 1 and nx.degree(g, longest_path[-1]) == 1:
            c.append(longest_path[0])
            c.append(longest_path[-1])
            r = max(rad[c[0]], rad[c[1]])
            for b in c:
                if rad[b] == r:
                    rootN = b
                    save_RootBranch.append(rootN)


        elif nx.degree(g, longest_path[0]) != 1 or nx.degree(g, longest_path[-1]) != 1:
            m = max(rad[longest_path[0]], rad[longest_path[-1]])
            for key, value in rad.items():
                if value == m:
                    if key == u or key == v:
                        rootN = key
                        save_RootBranch.append(rootN)
                        print("##### ROOTN######")
                        break


        directed_graph = nx.DiGraph()
        # # Add nodes to the directed graph
        # for node in dag.nodes():
        #     DG.add_node(node)
        #
        # # Add directed edges based on the root branch
        # for edge in dag.edges():
        #     if rootN in edge:
        #         source, target = edge
        #         DG.add_edge(source, target)
        #     else:
        #         target, source = edge
        #         DG.add_edge(source, target)





        # def add_directed_edges(node, parent, visited_nodes):
        #     neighbors = dag.neighbors(node)
        #     for neighbor in neighbors:
        #         if neighbor != parent:
        #             directed_graph.add_edge(node, neighbor)
        #             visited_nodes.add(neighbor)
        #             add_directed_edges(neighbor, node, visited_nodes)
        def add_directed_edges(node, parent, visited_nodes):
            stack = [(node, parent)]
            while stack:
                current_node, current_parent = stack.pop()
                neighbors = dag.neighbors(current_node)
                for neighbor in neighbors:
                    if neighbor != current_parent and neighbor not in visited_nodes:
                        directed_graph.add_edge(current_node, neighbor)
                        visited_nodes.add(neighbor)
                        stack.append((neighbor, current_node))


        # Create a set to track visited nodes
        # visited_nodes = set()

        # Specify the root branch
        root_branch = rootN
        # Add the root branch to the directed graph and visited nodes
        directed_graph.add_node(root_branch)
        # visited_nodes.add(root_branch)
        visited_nodes = set([root_branch])
        # Start adding directed edges from the root branch
        add_directed_edges(root_branch, None, visited_nodes)

        for source, target in directed_graph.edges():
            # Assuming edge attributes are stored as dictionaries in the main graph
            edge_attributes = dag[source][target]

            # Transfer attributes to the separate graph's edge
            directed_graph[source][target].update(edge_attributes)




        # f = list(g.edges(b))
        # for u, v, attr in g.edges(data=True):
        #     if u == f[0][0] or v == f[0][0]:
        #         attr['diameter'] = g[u][v]['diameter']

        # for i in range(0, len(points), 3):
        #     if points[i] == f[0][0] and points[i + 3] == f[0][1]:
        #         diameter = points[i + 2]
        #         attr['diameter'] == diameter


        # targetNode.extend([u, v])
        # if nx.degree(g, u) == 1:
        #     rootN = u
        # elif nx.degree(g, v) == 1:
        #     rootN = v
        # elif nx.degree(g, u) != 1 and nx.degree(g, v) != 1:
        #
        #     m = max(rad[u], rad[v])
        #     for key, value in rad.items():
        #         if value == m:
        #             if key == u or key == v:
        #                  rootN = key
        #                  print("##### ROOTN######")
        #                  break



        bfs_tree = nx.bfs_tree(directed_graph, rootN, reverse=False, depth_limit=None, sort_neighbors=True)
        parent = {node: None for node in bfs_tree.nodes()}
        for edge in bfs_tree.edges():
            parent[edge[1]] = edge[0]

        check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
        node_list = list(bfs_tree.nodes())
        edge_list = list(bfs_tree.edges())

        check = []
        ratiocheck = {}
        Diamcheck = False
        for node in bfs_tree.nodes():

            node = str(node)
            # node_index = node_list.index(node)
            node_index = list(bfs_tree.nodes()).index(node)
            if check_node[node_index] == False:

                if parent[node] != None:

                        par = directed_graph[parent[node]][node]["diameter"]
                    # if par != 0:
                        if len(list(directed_graph.neighbors(node))) > 0:
                            nnb = list(directed_graph.neighbors(node))

                            # if str(rootN) in nnb:
                            #     nnb.remove(str(rootN))
                            queue = []
                            for j in range(len(nnb)):
                                nnb_index = list(bfs_tree.nodes()).index(nnb[j])
                                if check_node[nnb_index] == False:
                                    if par!= 0:

                                        # children = nnb
                                        child = directed_graph[node][nnb[j]]["diameter"]
                                        ratioDiam = (child / par)

                                        check_node[node_index] = True
                                        check.append(node)
                                        # node_index = node_list.index(nnb[j])

                                        # check_node[nnb_index] = True

                                        directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam


                                        a = np.array(ast.literal_eval(node)) - np.array(ast.literal_eval(parent[node]))
                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(ast.literal_eval(node))
                                        inner = np.inner(a, b)
                                        norms = LA.norm(a) * LA.norm(b)
                                        cos = inner / norms
                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                        # rad = math.acos(cos)
                                        angle = np.rad2deg(radianc)
                                        directed_graph[node][nnb[j]]["angle"] = angle

                                    else:
                                        ratioDiam = -3
                                        node_index = list(bfs_tree.nodes()).index(node)
                                        check_node[node_index] = True
                                        directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam
                                        check.append(node)

                                        a = np.array(ast.literal_eval(node)) - np.array(
                                            ast.literal_eval(parent[node]))
                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                            ast.literal_eval(node))
                                        inner = np.inner(a, b)
                                        norms = LA.norm(a) * LA.norm(b)
                                        cos = inner / norms
                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                        # rad = math.acos(cos)
                                        angle = np.rad2deg(radianc)
                                        directed_graph[node][nnb[j]]["angle"] = angle

                                # elif check_node[nnb_index] == True and parent[nnb[j]] == None and len(nnb) == 1: # seprate branch
                                #     break

                                elif check_node[nnb_index] == True and node not in check: # terminal branch

                                    # for edge in edge_list:
                                    #     if node and nnb[j] in edge:
                                            if "ratioDiam" in directed_graph[node][nnb[j]]:

                                                check_node[node_index] = True
                                            else:
                                                child = directed_graph[node][nnb[j]]["diameter"]
                                                if par!=0:
                                                    ratioDiam = (child / par)
                                                    directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                    a = np.array(ast.literal_eval(node)) - np.array(
                                                        ast.literal_eval(parent[node]))
                                                    b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                        ast.literal_eval(node))
                                                    inner = np.inner(a, b)
                                                    norms = LA.norm(a) * LA.norm(b)
                                                    cos = inner / norms
                                                    radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                    # rad = math.acos(cos)
                                                    angle = np.rad2deg(radianc)
                                                    directed_graph[node][nnb[j]]["angle"] = angle


                                                else:
                                                    ratioDiam = -3
                                                    directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                    a = np.array(ast.literal_eval(node)) - np.array(
                                                        ast.literal_eval(parent[node]))
                                                    b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                        ast.literal_eval(node))
                                                    inner = np.inner(a, b)
                                                    norms = LA.norm(a) * LA.norm(b)
                                                    cos = inner / norms
                                                    radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                    # rad = math.acos(cos)
                                                    angle = np.rad2deg(radianc)
                                                    directed_graph[node][nnb[j]]["angle"] = angle



                else:

                    ratioDiam = -4
                    node_index = list(bfs_tree.nodes()).index(node)
                    check_node[node_index] = True
                    nnb = list(directed_graph.neighbors(node))
                    directed_graph[node][nnb[0]]["ratioDiam"] = ratioDiam
                    check.append(node)
                    angle = -4
                    directed_graph[node][nnb[0]]["angle"] = angle

        separate_directed_graphs.add_nodes_from(directed_graph.nodes(data=True))
        separate_directed_graphs.add_edges_from(directed_graph.edges(data=True))
        # separate_directed_graphs.subgraph()

    return separate_directed_graphs, save_RootBranch


def Final_Finalgraph(graph, Roots):
    g = graph.copy()



    for u, v in g.edges:


        if 'diameter' not in g[u][v]:
            print("######################this branch doesnt have diameter######################")
            g[u][v]['diameter'] = 0.8

            g[u][v]['length'] = 1.5


        else:
            diameter = g[u][v]['diameter']
            length = g[u][v]['length']


    for b in range(len(Roots)):
        Root_points = Roots[b]
        for rootN in Root_points:

            bfs_tree = nx.bfs_tree(g, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)

            # Find the parent of each node
            parent = {}
            for node in bfs_tree.nodes():
                parent_nodes = list(bfs_tree.predecessors(node))
                if parent_nodes:
                    parent[node] = parent_nodes[0]
                else:
                    parent[node] = None
            for edge in bfs_tree.edges():
                parent[edge[1]] = edge[0]

            check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
            node_list = list(bfs_tree.nodes())
            edge_list = list(bfs_tree.edges())

            check = []
            ratiocheck = {}
            Diamcheck = False
            for node in bfs_tree.nodes():

                node = str(node)
                # node_index = node_list.index(node)
                node_index = list(bfs_tree.nodes()).index(node)
                if check_node[node_index] == False:

                    if parent[node] != None:

                            par = g[parent[node]][node]["diameter"]
                        # if par != 0:
                            if len(list(g.neighbors(node))) > 0:
                                nnb = list(g.neighbors(node))

                                # if str(rootN) in nnb:
                                #     nnb.remove(str(rootN))
                                queue = []
                                for j in range(len(nnb)):
                                    nnb_index = list(bfs_tree.nodes()).index(nnb[j])
                                    if check_node[nnb_index] == False:
                                        if par!= 0:

                                            # children = nnb
                                            child = g[node][nnb[j]]["diameter"]
                                            ratioDiam = (child / par)

                                            check_node[node_index] = True
                                            check.append(node)
                                            # node_index = node_list.index(nnb[j])

                                            # check_node[nnb_index] = True

                                            g[node][nnb[j]]["ratioDiam"] = ratioDiam


                                            a = np.array(ast.literal_eval(node)) - np.array(ast.literal_eval(parent[node]))
                                            b = np.array(ast.literal_eval(nnb[j])) - np.array(ast.literal_eval(node))
                                            inner = np.inner(a, b)
                                            norms = LA.norm(a) * LA.norm(b)
                                            cos = inner / norms
                                            radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                            # rad = math.acos(cos)
                                            angle = np.rad2deg(radianc)
                                            g[node][nnb[j]]["angle"] = angle

                                        else:
                                            ratioDiam = -3
                                            node_index = list(bfs_tree.nodes()).index(node)
                                            check_node[node_index] = True
                                            g[node][nnb[j]]["ratioDiam"] = ratioDiam
                                            check.append(node)

                                            a = np.array(ast.literal_eval(node)) - np.array(
                                                ast.literal_eval(parent[node]))
                                            b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                ast.literal_eval(node))
                                            inner = np.inner(a, b)
                                            norms = LA.norm(a) * LA.norm(b)
                                            cos = inner / norms
                                            radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                            # rad = math.acos(cos)
                                            angle = np.rad2deg(radianc)
                                            g[node][nnb[j]]["angle"] = angle

                                    # elif check_node[nnb_index] == True and parent[nnb[j]] == None and len(nnb) == 1: # seprate branch
                                    #     break

                                    elif check_node[nnb_index] == True and node not in check: # terminal branch

                                        # for edge in edge_list:
                                        #     if node and nnb[j] in edge:
                                                if "ratioDiam" in g[node][nnb[j]]:

                                                    check_node[node_index] = True
                                                else:
                                                    child = g[node][nnb[j]]["diameter"]
                                                    if par!=0:
                                                        ratioDiam = (child / par)
                                                        g[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                        a = np.array(ast.literal_eval(node)) - np.array(
                                                            ast.literal_eval(parent[node]))
                                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                            ast.literal_eval(node))
                                                        inner = np.inner(a, b)
                                                        norms = LA.norm(a) * LA.norm(b)
                                                        cos = inner / norms
                                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                        # rad = math.acos(cos)
                                                        angle = np.rad2deg(radianc)
                                                        g[node][nnb[j]]["angle"] = angle


                                                    else:
                                                        ratioDiam = -3
                                                        g[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                        a = np.array(ast.literal_eval(node)) - np.array(
                                                            ast.literal_eval(parent[node]))
                                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                            ast.literal_eval(node))
                                                        inner = np.inner(a, b)
                                                        norms = LA.norm(a) * LA.norm(b)
                                                        cos = inner / norms
                                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                        # rad = math.acos(cos)
                                                        angle = np.rad2deg(radianc)
                                                        g[node][nnb[j]]["angle"] = angle



                    else:

                        ratioDiam = -4
                        node_index = list(bfs_tree.nodes()).index(node)
                        check_node[node_index] = True
                        nnb = list(g.neighbors(node))
                        g[node][nnb[0]]["ratioDiam"] = ratioDiam
                        check.append(node)
                        angle = -4
                        g[node][nnb[0]]["angle"] = angle


            # separate_directed_graphs.subgraph()

    return g




from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
def joint_graphs_mainArtery(graph, radi, RootBranches):
    g = graph.copy()
    # main_graph = g

    # components = nx.connected_components(g)
    components = list(nx.weakly_connected_components(g))

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    subgraphs = []

    main_graph = g

    rootNode = []
    MiddlePoint=[]
    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]

        degreeStore=[]
        rootBranch = []
        subgraph= nx.Graph()
        subgraph.add_nodes_from(nodes_in_label)
        subgraph.add_edges_from(edges_in_label)

        node_positions={}
        for i in range(len(nodes_in_label)):
            node_positions[nodes_in_label[i]] =  tuple(ast.literal_eval(nodes_in_label[i]))

        nx.set_node_attributes(subgraph, node_positions, 'pos')



        subgraphs.append(subgraph)

        main_graph = nx.compose(main_graph, subgraph)



        store_diameter = {}
        for u1, v1 in subgraph.edges():
            # if 'radius' in data:
                diameter = main_graph[u1][v1]['diameter']
                store_diameter[(u1, v1)] = diameter
        sorted_store_diameter =  sorted(store_diameter.items(), key=lambda item: item[1], reverse=True)
        max_radius_edge, max_radius_value = max(store_diameter.items(), key=lambda item: item[1])

        rootBranch.append(max_radius_edge)
        nodes_to_check = [node for edge in rootBranch for node in edge]
        for node_name in nodes_to_check:
            if node_name in main_graph:
                degree = main_graph.degree[node_name]
                degreeStore.append(degree)

                if degree == 1 or degree == 2:
                    rootNode.append(node_name)
                    break
        if degreeStore[0]>=3 and degreeStore[1]>=3:
             max_radius_value = sorted_store_diameter[1][1]
             max_radius_edge = sorted_store_diameter[1][0]
             rootBranch.clear()
             nodes_to_check.clear()
             degreeStore.clear()
             rootBranch.append(max_radius_edge)
             nodes_to_check = [node for edge in rootBranch for node in edge]
             for node_name in nodes_to_check:
                if node_name in main_graph:
                    degree = main_graph.degree[node_name]
                    degreeStore.append(degree)
                    if degree == 1 or degree == 2:
                        rootNode.append(node_name)
                        break




    def str_to_tuple(s):
        if isinstance(s, str):
            # Remove the square brackets and split by commas
            parts = s.strip('[]').split(',')
            # Convert parts to integers and create a tuple
            return tuple(float(x.strip()) for x in parts)
        else:
            # Handle non-string inputs, such as already converted tuples
            return s
    # Create a dictionary with keys 'point1', 'point2', etc.
    points = {
        f'point{i}': str_to_tuple(s)
        for i, s in enumerate(RootBranches, start=1)
        }
    print(points)


    # points = rootNode
    # points = {f'point{i + 1}': tuple(ast.literal_eval(coord)) for i, coord in enumerate(rootNode)}




    # # Convert points to a numpy array for DBSCAN
    # point_coords = np.array(list(points.values()))
    # # Create a DBSCAN clusterer
    # dbscan = DBSCAN(eps=5.0, min_samples=2)  # Adjust eps and min_samples as needed
    # # Fit the clusterer to your data
    # labels = dbscan.fit_predict(point_coords)
    # # Create a dictionary to store node labels
    # node_labels = {}
    # # Assign labels to nodes based on the DBSCAN clusters
    # for node, label in zip(points.keys(), labels):
    #     node_labels[node] = label + 1  # Adding 1 to start labeling from 1
    # # Create a dictionary to store clusters
    # clusters = defaultdict(list)
    # # Group points by cluster label
    # for node, label in node_labels.items():
    #     clusters[label].append(node)
    # # Print the points in each cluster
    # for label, cluster in clusters.items():
    #     print(f"Cluster {label}: {cluster}")



    # Convert points to a numpy array for clustering
    point_coords = np.array(list(points.values()))
    # point_coords = np.array([eval(coord) for coord in points])
    # Perform hierarchical clustering
    linked = linkage(point_coords, method='ward')  # You can try different linkage methods

    labels = fcluster(linked, t=93, criterion='distance')  # Adjust 't' value as needed
    labels -= 1
    # Create a dictionary to store node labels
    node_labels = {}
    # Assign labels to nodes based on the clustering result
    for node, label in zip(points.keys(), labels):
        node_labels[node] = label
    # for i, label in enumerate(labels):
    #     node_labels[f'point{i+1}'] = label
    # Create a dictionary to store clusters
    clusters = defaultdict(list)
    # Group points by cluster label
    for node, label in node_labels.items():
        clusters[label].append(node)
    # Print the points in each cluster
    for label, cluster in clusters.items():
        print(f"Cluster {label}: {cluster}")


    # each_point = []
    # for k in range(len(clusters)+1):
    #     if len(clusters[k]) != 0:
    #         for index, point_key in enumerate(clusters[k]):
    #             each_point.append(points[clusters[k][index]])


    for k in range(len(clusters)):
            each_point = []
            # for index, point_key in enumerate(clusters[k]):
            #     each_point.append(eval([clusters[k][index]]))
            # for point_key in clusters[0][k]:
            #     each_point.append(eval(point_key))

            each_point = []
            for point_key in clusters[k]:
                # Check if the point_key exists in the 'points' dictionary
                if point_key in points:
                    each_point.append(points[point_key])

            # Filter out any non-tuple elements (e.g., string point keys)
            each_point = [point for point in each_point if isinstance(point, tuple)]

            if len(each_point) == 0:
                # Handle cases where there are no valid coordinates
                continue


            # each_point = [item for sublist in each_point for item in sublist]
            # Calculate the average of each coordinate (x, y, z)
            num_points = len(each_point)
            middle_x = sum(point[0] for point in each_point) / num_points
            middle_y = sum(point[1] for point in each_point) / num_points
            middle_z = sum(point[2] for point in each_point) / num_points


            # The resulting coordinates represent the point in the middle of the four points
            middle_point = (middle_x, middle_y, middle_z)
            middle_point = list(middle_point)
            MiddlePoint.append(middle_point)
            Middle_Point = Vertex(middle_point)
            # aa.clear()

            main_graph.add_node(str(Middle_Point), pos=Middle_Point.point)

            for point in clusters[k]:
                point = list(points[point])
                main_graph.add_edge(str(middle_point), str((point)))

    average_Middle_points_X = sum(point[0] for point in MiddlePoint) / len(MiddlePoint)
    average_Middle_points_y = sum(point[1] for point in MiddlePoint) / len(MiddlePoint)
    average_Middle_points_z = sum(point[2] for point in MiddlePoint) / len(MiddlePoint)

    Coor_Middle_point = (average_Middle_points_X, average_Middle_points_y, average_Middle_points_z)
    middle_middle_points = list(Coor_Middle_point)
    Middle_Middle_Points = Vertex(middle_middle_points)
    main_graph.add_node(str(Middle_Middle_Points), pos=Middle_Middle_Points.point)

    for index, Midpoint in enumerate(MiddlePoint):
        main_graph.add_edge(str(middle_middle_points), str((Midpoint)))


    # predicting the end point for main pulmonary artery
    main_artery_len = 40
    middle_middle_end_point = (middle_middle_points[0] , middle_middle_points[1] , middle_middle_points[2]  - main_artery_len)
    middle_middle_end_point = list(middle_middle_end_point)
    Middle_Middle_End_point = Vertex(middle_middle_end_point)
    main_graph.add_node(str(Middle_Middle_End_point), pos=Middle_Middle_End_point.point)
    main_graph.add_edge(str(middle_middle_end_point), str((Midpoint)))

            # each_point.clear()

            # # Convert the strings to numeric tuples
            # point_a = str_to_tuple(middle_point)
            # point_b = list(str_to_tuple(each_point[k]))
            #
            # # Calculate the Euclidean distance
            # distance = np.linalg.norm(np.array(point_a) - np.array(point_b))
            # main_graph[str(middle_point)][str(list(each_point[k]))]["length"] = distance #+ endNodeAddition
            # main_graph[str(middle_point)][str(list(each_point[k]))]["radius"] = radi[str(list(each_point[k]))]
            # main_graph[str(middle_point)][str(list(each_point[k]))]["diameter"] = 2*radi[str(list(each_point[k]))]
            # main_graph[str(middle_point)][str(list(each_point[k]))]["RatioLendiameter"] = distance / (2*radi[str(list(each_point[k]))])


        #
        # length = 10.0
        # direction_vector = np.array([-1.0, 1.0, -1.0])  # Change this vector for a different direction
        # unit_vector = direction_vector / np.linalg.norm(direction_vector)
        # end_point = np.array([middle_point[0], middle_point[1], middle_point[2]]) + length * unit_vector
        #
        # # direction_vector = np.array([middle_point[0], middle_point[1], middle_point[2]])
        # # unit_vector = direction_vector / np.linalg.norm(direction_vector)
        #
        # # # Calculate the endpoint
        # # x_end = middle_point[0] + unit_vector[0] * length
        # # y_end = middle_point[1] + unit_vector[1] * length
        # # z_end = middle_point[2] + unit_vector[2] * length
        #
        # end_point = list(end_point)
        # End_Point = Vertex(end_point)
        # main_graph.add_node(str(End_Point), pos=End_Point.point)
        # main_graph.add_edge(str(middle_point), str(end_point))
        #
        # # Convert the strings to numeric tuples
        # pointa = str_to_tuple(middle_point)
        # pointb = list(str_to_tuple(end_point))
        #
        # # Calculate the Euclidean distance
        # distance = np.linalg.norm(np.array(pointa) - np.array(pointb))
        # main_graph[str(middle_point)][str(list(end_point))]["length"] = distance #+ endNodeAddition
        # main_graph[str(middle_point)][str(list(end_point))]["radius"] = radi[str(list(each_point[k]))]
        # main_graph[str(middle_point)][str(list(end_point))]["diameter"] = 2*radi[str(list(each_point[k]))]
        # main_graph[str(middle_point)][str(list(end_point))]["RatioLendiameter"] = distance / (radi[str(list(each_point[k]))])



    return main_graph



def Final_graph_Group2(graph, rad):
    gD = graph.copy()
    g = gD.to_undirected()
    components = nx.connected_components(g)

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    label_root_store = {}
    separate_directed_graphs = nx.DiGraph()
    save_RootBranch = []
    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]

        points = []
        radiiii = []
        lengths = []
        label_branches = defaultdict(list)
        for u, v in edges_in_label:

            points.append(u)
            points.append(v)
            if 'diameter' not in g[u][v]:
                print("######################this branch doest have diameter######################")
                g[u][v]['diameter'] = 0.8
                g[u][v]['length'] = 1.5
                radiiii.append(diameter)
                lengths.append(length)
                label_branches[label_array[i]].append((diameter, length))
            else:
                diameter = g[u][v]['diameter']
                length = g[u][v]['length']
                points.append(diameter)
                radiiii.append(diameter)
                lengths.append(length)
                label_branches[label_array[i]].append((diameter, length))

        radiiii.sort(reverse = True)


        dag = nx.Graph()
        dag.add_nodes_from(nodes_in_label)
        dag.add_edges_from(edges_in_label)
        for source, target in dag.edges():
            # Assuming edge attributes are stored as dictionaries in the main graph
            edge_attributes = g[source][target]

            # Transfer attributes to the separate graph's edge
            dag[source][target].update(edge_attributes)


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




        path_distances = []
        if len(all_paths) != 0:
            for path in all_paths:
                distance = 0
                for i in range(len(path) - 1):
                    distance += euclidean_distance(path[i], path[i + 1])
                path_distances.append((path, distance))

            # Sort the paths by distance in descending order
            path_distances.sort(key=lambda x: x[1], reverse=True)

            # Print the longest path
            longest_path, longest_distance = path_distances[0]
            print("Longest Path:", longest_path)
            print("Distance:", longest_distance)
            c =[]
        else:
            print("################ there is no path############################")
            g.remove_nodes_from(dag.nodes())
            g.remove_edges_from(dag.edges())

            continue

        if  nx.degree(g, longest_path[0]) == 1 and nx.degree(g, longest_path[-1]) == 1:
            c.append(longest_path[0])
            c.append(longest_path[-1])
            r = max(rad[c[0]], rad[c[1]])
            for b in c:
                if rad[b] == r:
                    rootN = b
                    save_RootBranch.append(rootN)

        elif nx.degree(g, longest_path[0]) != 1 or nx.degree(g, longest_path[-1]) != 1:
            m = max(rad[longest_path[0]], rad[longest_path[-1]])
            for key, value in rad.items():
                if value == m:
                    if key == u or key == v:
                        rootN = key
                        save_RootBranch.append(rootN)
                        print("##### ROOTN######")
                        break


        directed_graph = nx.DiGraph()

        def add_directed_edges(node, parent, visited_nodes):
            neighbors = dag.neighbors(node)
            for neighbor in neighbors:
                if neighbor != parent:
                    directed_graph.add_edge(node, neighbor)
                    visited_nodes.add(neighbor)
                    add_directed_edges(neighbor, node, visited_nodes)




        # Specify the root branch
        root_branch = rootN
        # Add the root branch to the directed graph and visited nodes
        directed_graph.add_node(root_branch)
        # visited_nodes.add(root_branch)
        visited_nodes = set([root_branch])
        # Start adding directed edges from the root branch
        add_directed_edges(root_branch, None, visited_nodes)

        for source, target in directed_graph.edges():
            # Assuming edge attributes are stored as dictionaries in the main graph
            edge_attributes = dag[source][target]

            # Transfer attributes to the separate graph's edge
            directed_graph[source][target].update(edge_attributes)



        bfs_tree = nx.bfs_tree(directed_graph, rootN, reverse=False, depth_limit=None, sort_neighbors=True)
        parent = {node: None for node in bfs_tree.nodes()}
        for edge in bfs_tree.edges():
            parent[edge[1]] = edge[0]

        check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
        node_list = list(bfs_tree.nodes())
        edge_list = list(bfs_tree.edges())

        check = []
        ratiocheck = {}
        Diamcheck = False
        for node in bfs_tree.nodes():

            node = str(node)
            # node_index = node_list.index(node)
            node_index = list(bfs_tree.nodes()).index(node)
            if check_node[node_index] == False:

                if parent[node] != None:

                        par = directed_graph[parent[node]][node]["diameter"]
                    # if par != 0:
                        if len(list(directed_graph.neighbors(node))) > 0:
                            nnb = list(directed_graph.neighbors(node))

                            # if str(rootN) in nnb:
                            #     nnb.remove(str(rootN))
                            queue = []
                            for j in range(len(nnb)):
                                nnb_index = list(bfs_tree.nodes()).index(nnb[j])
                                if check_node[nnb_index] == False:
                                    if par!= 0:

                                        # children = nnb
                                        child = directed_graph[node][nnb[j]]["diameter"]
                                        ratioDiam = (child / par)

                                        check_node[node_index] = True
                                        check.append(node)
                                        # node_index = node_list.index(nnb[j])

                                        # check_node[nnb_index] = True

                                        directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam


                                        a = np.array(ast.literal_eval(node)) - np.array(ast.literal_eval(parent[node]))
                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(ast.literal_eval(node))
                                        inner = np.inner(a, b)
                                        norms = LA.norm(a) * LA.norm(b)
                                        cos = inner / norms
                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                        # rad = math.acos(cos)
                                        angle = np.rad2deg(radianc)
                                        directed_graph[node][nnb[j]]["angle"] = angle

                                    else:
                                        ratioDiam = -3
                                        node_index = list(bfs_tree.nodes()).index(node)
                                        check_node[node_index] = True
                                        directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam
                                        check.append(node)

                                        a = np.array(ast.literal_eval(node)) - np.array(
                                            ast.literal_eval(parent[node]))
                                        b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                            ast.literal_eval(node))
                                        inner = np.inner(a, b)
                                        norms = LA.norm(a) * LA.norm(b)
                                        cos = inner / norms
                                        radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                        # rad = math.acos(cos)
                                        angle = np.rad2deg(radianc)
                                        directed_graph[node][nnb[j]]["angle"] = angle

                                # elif check_node[nnb_index] == True and parent[nnb[j]] == None and len(nnb) == 1: # seprate branch
                                #     break

                                elif check_node[nnb_index] == True and node not in check: # terminal branch

                                    # for edge in edge_list:
                                    #     if node and nnb[j] in edge:
                                            if "ratioDiam" in directed_graph[node][nnb[j]]:

                                                check_node[node_index] = True
                                            else:
                                                child = directed_graph[node][nnb[j]]["diameter"]
                                                if par!=0:
                                                    ratioDiam = (child / par)
                                                    directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                    a = np.array(ast.literal_eval(node)) - np.array(
                                                        ast.literal_eval(parent[node]))
                                                    b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                        ast.literal_eval(node))
                                                    inner = np.inner(a, b)
                                                    norms = LA.norm(a) * LA.norm(b)
                                                    cos = inner / norms
                                                    radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                    # rad = math.acos(cos)
                                                    angle = np.rad2deg(radianc)
                                                    directed_graph[node][nnb[j]]["angle"] = angle


                                                else:
                                                    ratioDiam = -3
                                                    directed_graph[node][nnb[j]]["ratioDiam"] = ratioDiam

                                                    a = np.array(ast.literal_eval(node)) - np.array(
                                                        ast.literal_eval(parent[node]))
                                                    b = np.array(ast.literal_eval(nnb[j])) - np.array(
                                                        ast.literal_eval(node))
                                                    inner = np.inner(a, b)
                                                    norms = LA.norm(a) * LA.norm(b)
                                                    cos = inner / norms
                                                    radianc = np.arccos(np.clip(cos, -1.0, 1.0))
                                                    # rad = math.acos(cos)
                                                    angle = np.rad2deg(radianc)
                                                    directed_graph[node][nnb[j]]["angle"] = angle





                else:

                    ratioDiam = -4
                    node_index = list(bfs_tree.nodes()).index(node)
                    check_node[node_index] = True
                    nnb = list(directed_graph.neighbors(node))
                    directed_graph[node][nnb[0]]["ratioDiam"] = ratioDiam
                    check.append(node)
                    angle = -4
                    directed_graph[node][nnb[0]]["angle"] = angle

        separate_directed_graphs.add_nodes_from(directed_graph.nodes(data=True))
        separate_directed_graphs.add_edges_from(directed_graph.edges(data=True))

    return separate_directed_graphs, save_RootBranch

def graph_with_side_branches_group1(graph, sepBranch, graphwithnodes):

    g = graph.copy()
    g1 = graphwithnodes.copy()


    # sepBranch = [([eval(item[0]), eval(item[1])]) for item in sepBranch]
    x = []

    avglen = []
    avgratiolendiam = []
    for u, v, o in g.edges(data="object"):

        if g[u][v]["RatioLendiameter"] <= 5:
             avgratiolendiam.append(g[u][v]["RatioLendiameter"])
             avglen.append(g[u][v]["length"])

    avg_ratioLenDiam = statistics.mean(avgratiolendiam)
    avg_ratioLenDiam = math.ceil(avg_ratioLenDiam)
    avg_len = statistics.mean(avglen)
    avg_len = math.ceil(avg_len)


    for  u, v, o in g.edges(data="object"):
        if g[u][v]["RatioLendiameter"] > 5:

            # for u, v, o in g.edges(data="object"):
                allpoints = []
                p = {}
                allpoints.append(u)
                allpoints.append(o.pixels)
                allpoints.append(v)
                flattened_list = [item if isinstance(item, str) else item[0] for sublist in allpoints for item in (sublist if isinstance(sublist, list) else [sublist])]
                for i in range(len(sepBranch)):
                    for j in range(len(flattened_list)):

                        a = math.sqrt((np.array(ast.literal_eval(sepBranch[i][0]))[0] - ast.literal_eval(flattened_list[j])[0])**2 + (np.array(ast.literal_eval(sepBranch[i][0]))[1] - ast.literal_eval(flattened_list[j])[1])**2 + (np.array(ast.literal_eval(sepBranch[i][0]))[2] - ast.literal_eval(flattened_list[j])[2])**2)
                        b = math.sqrt((np.array(ast.literal_eval(sepBranch[i][1]))[0] - ast.literal_eval(flattened_list[j])[0])**2 + (np.array(ast.literal_eval(sepBranch[i][1]))[1] - ast.literal_eval(flattened_list[j])[1])**2 + (np.array(ast.literal_eval(sepBranch[i][1]))[2] - ast.literal_eval(flattened_list[j])[2])**2)
                        # a = LA.norm(np.array(ast.literal_eval(sepBranch[i][0]))) - LA.norm(np.array(ast.literal_eval(flattened_list[j])))
                        # b = LA.norm(np.array(ast.literal_eval(sepBranch[i][1]))) - LA.norm(np.array(ast.literal_eval(flattened_list[j])))

                        p[(sepBranch[i][0],flattened_list[j])] = LA.norm(a)
                        p[(sepBranch[i][1],flattened_list[j])] = LA.norm(b)
                        # p.append(LA.norm(a))
                        # p.append(LA.norm(b))
                values_array = np.array(list(p.values()))
                min_indices = np.argsort(values_array)[:3]
                # Extract the minimum values using the indices
                mindistance = values_array[min_indices]
                # mindistance = min(values_array)
                found_key = []
                for key, value in p.items():
                    for n in range(len(mindistance)):
                        if mindistance[n] < 3 and np.allclose(value, mindistance[n]):
                                found_key.append(key)


                if len(found_key) != 0:
                    for l in range(len(found_key)):
                        # if mindistance[l] < 3:
                            if found_key[l][0] not in x and found_key[l][1] not in x:
                                p1 = ast.literal_eval(found_key[l][0])
                                p2 = ast.literal_eval(found_key[l][1])
                                p3 = ast.literal_eval(v)
                                direction1 = np.array(p2) - np.array(p1)
                                direction2 = np.array(p3) - np.array(p1)
                                norm_direction1 = direction1 / np.linalg.norm(direction1)
                                norm_direction2 = direction2 / np.linalg.norm(direction2)
                                cosine_angle = np.dot(norm_direction1, norm_direction2)
                                degree_angle = abs(math.degrees(cosine_angle))
                                angle = 50

                                if degree_angle <= angle:

                                    x.append(found_key[l][0])
                                    x.append(found_key[l][1])
                                    currV = Vertex(found_key[l][0])
                                    new_point = Vertex(found_key[l][1])

                                    g1.add_edge(currV.point, new_point.point, object=Edge(currV, new_point))


                # else:
                #
                #     branches = [(np.array(ast.literal_eval(u)), np.array(ast.literal_eval(v))),]
                #
                #
                #     newlen = g[u][v]["length"] / avg_len
                #     newlen = math.ceil(newlen)
                #     divisions = newlen
                #
                #     new_branches = []
                #     for start_point, end_point in branches:
                #         length = np.linalg.norm(end_point - start_point)
                #         direction_vector = (end_point - start_point) / length
                #         division_lengths = np.linspace(0, length, divisions + 1)
                #
                #         for i in range(1, len(division_lengths)):
                #             new_start_point = list(start_point + division_lengths[i-1] * direction_vector)
                #             new_end_point = list(start_point + division_lengths[i] * direction_vector)
                #             new_branches.append((new_start_point, new_end_point))
                #
                #     # for i, (new_start, new_end) in enumerate(new_branches):
                #     #      formatted_start = ', '.join([f"{coord:.3f}" for coord in new_start])
                #     #      formatted_end = ', '.join([f"{coord:.3f}" for coord in new_end])
                #     #      print(f"Branch {i+1} - Start: [{formatted_start}], End: [{formatted_end}]")
                #
                #     for w in range(len(new_branches)):
                #         add_more_newbranches = []
                #         if len(new_branches) > 2:
                #             if not str((new_branches[w][0])) in g.nodes() and not str((new_branches[w][1])) in g.nodes():
                #                 startpoint = Vertex(((new_branches[w][0])))
                #                 endpoint = Vertex(((new_branches[w][1])))
                #                 g1.add_node(str(startpoint.point), pos=startpoint.point)
                #                 g1.add_node(str(endpoint.point), pos=endpoint.point)
                #                 g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                #
                #
                #
                #                 if w % 2 == 0:
                #
                #                     half_main_branch_length = (np.linalg.norm(np.array(new_branches[w][1]) - np.array(new_branches[w][0]))) / 2
                #                     main_direction_vector = (np.array(new_branches[w][1]) - np.array(new_branches[w][0])) / half_main_branch_length
                #
                #                     # Calculate the new branch start and end points
                #                     new_branch_start = new_branches[w][0]
                #                     new_branch_end = new_branches[w][0] + half_main_branch_length * main_direction_vector
                #
                #                     # Calculate the angle in radians for the 40-degree rotation
                #                     angle_degrees = 40.0
                #                     angle_radians = np.radians(angle_degrees)
                #
                #                     #Calculate the rotation matrix for the 40-degree rotation around the main direction vector
                #                     rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)], [0, 1, 0], [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
                #
                #                     # Calculate the new branch end point with the 40-degree rotation
                #                     new_branch_end_rotated = new_branch_end + rotation_matrix @ main_direction_vector
                #                     newpoint = Vertex((list(new_branch_end_rotated)))
                #                     g1.add_node(str(newpoint.point), pos=newpoint.point)
                #                     g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))
                #                     g1[str(startpoint.point)][str(newpoint.point)]['radius'] = (g[u][v]['radius']) / 2
                #
                #                     # g1[startpoint.point][newpoint.point]['length'] = half_main_branch_length
                #
                #
                #
                #                     # Print the new branch coordinates
                #                     print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                #                 else:
                #                     half_main_branch_length = (np.linalg.norm(np.array(new_branches[w][1]) - np.array(new_branches[w][0]))) / 2
                #                     main_direction_vector = (np.array(new_branches[w][1]) - np.array(new_branches[w][0])) / half_main_branch_length
                #
                #                     # Calculate the new branch start and end points
                #                     new_branch_start = new_branches[w][0]
                #                     new_branch_end = new_branches[w][0] + half_main_branch_length * main_direction_vector
                #
                #                     # Calculate the angle in radians for the 40-degree rotation
                #                     angle_degrees = -40.0
                #                     angle_radians = np.radians(angle_degrees)
                #
                #                     #Calculate the rotation matrix for the 40-degree rotation around the main direction vector
                #                     rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)], [0, 1, 0], [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
                #
                #                     # Calculate the new branch end point with the 40-degree rotation
                #                     new_branch_end_rotated = new_branch_end + rotation_matrix @ main_direction_vector
                #                     newpoint = Vertex((list(new_branch_end_rotated)))
                #                     g1.add_node(str(newpoint.point), pos=newpoint.point)
                #                     g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))
                #                     g1[str(startpoint.point)][str(newpoint.point)]['radius'] = (g[u][v]['radius']) / 2
                #
                #                     # Print the new branch coordinates
                #                     # network_plot_3D(g1, angle=0)
                #                     print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                #                     # network_plot_3D(g1, angle=0)
                #
                #             elif not str((new_branches[w][0])) in g.nodes() or not str((new_branches[w][1])) in g.nodes():
                #                 startpoint = Vertex(((new_branches[w][0])))
                #                 endpoint = Vertex(((new_branches[w][1])))
                #                 g1.add_node(str(startpoint.point), pos=startpoint.point)
                #                 g1.add_node(str(endpoint.point), pos=endpoint.point)
                #                 g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                #         if len(new_branches) == 2:
                #             if not str(list(new_branches[w][0])) in g.nodes() or not str(list(new_branches[w][1])) in g.nodes():
                #                 startpoint = Vertex((list(new_branches[w][0])))
                #                 endpoint = Vertex((list(new_branches[w][1])))
                #                 g1.add_node(str(startpoint.point), pos=startpoint.point)
                #                 g1.add_node(str(endpoint.point), pos=endpoint.point)
                #                 g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                #
                #                 half_main_branch_length = (np.linalg.norm(np.array(new_branches[w][1]) - np.array(new_branches[w][0]))) / 2
                #                 main_direction_vector = (np.array(new_branches[w][1]) - np.array(new_branches[w][0])) / half_main_branch_length
                #
                #                 # Calculate the new branch start and end points
                #                 new_branch_start = new_branches[w][0]
                #                 new_branch_end = new_branches[w][0] + half_main_branch_length * main_direction_vector
                #
                #                 # Calculate the angle in radians for the 40-degree rotation
                #                 angle_degrees = 40.0
                #                 angle_radians = np.radians(angle_degrees)
                #
                #                 #Calculate the rotation matrix for the 40-degree rotation around the main direction vector
                #                 rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)], [0, 1, 0], [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
                #
                #                 # Calculate the new branch end point with the 40-degree rotation
                #                 new_branch_end_rotated = new_branch_end + rotation_matrix @ main_direction_vector
                #                 newpoint = Vertex((list(new_branch_end_rotated)))
                #                 g1.add_node(str(newpoint.point), pos=newpoint.point)
                #                 g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))
                #                 g1[str(startpoint.point)][str(newpoint.point)]['radius'] = (g[u][v]['radius']) / 2
                #                 break
                #
                #
                #
                #             elif not str((new_branches[w][0])) in g.nodes() or not str((new_branches[w][1])) in g.nodes():
                #                 startpoint = Vertex(((new_branches[w][0])))
                #                 endpoint = Vertex(((new_branches[w][1])))
                #                 g1.add_node(str(startpoint.point), pos=startpoint.point)
                #                 g1.add_node(str(endpoint.point), pos=endpoint.point)
                #                 g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                #







    return g1

def Remove_Degree_BiggerThree(graph, rootNodes):
    g = graph.copy()
    components = list(nx.weakly_connected_components(g))

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    label_root_store = {}
    subgraph = nx.DiGraph()

    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]


        dag = nx.DiGraph()
        dag.add_nodes_from(nodes_in_label)
        dag.add_edges_from(edges_in_label)
        visitNode = []
        G = dag.copy()
        G1 = dag.copy()
        def process_node(node, visited):
            # Check if the node has already been visited
            if node in visited:
                return

            # Mark the current node as visited
            visited.add(node)

            visitNode.append(node)

            degree = G.degree(node)
            print(f'Degree of {node}: {degree}')
            if degree >3:

                def cluster_neighbors(neighbors):
                    # Initialize a list to save clusters
                    SaveCluster = []

                    # Convert the neighbor coordinates to a NumPy array
                    w = [list(map(float, element.strip('[]').split(','))) for element in neighbors]
                    w_array = np.array(w)

                    # Create a KMeans clustering model with 2 clusters
                    kmeans = KMeans(n_clusters=2, random_state=0)

                    # Fit the model to the data
                    kmeans.fit(w_array)

                    # Get cluster labels
                    labels = kmeans.labels_

                    # Separate the points into two clusters based on the labels
                    cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
                    cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]

                    # Append the clusters to SaveCluster
                    SaveCluster.append(cluster_1)
                    SaveCluster.append(cluster_2)

                    return SaveCluster

                def calculate_average_and_update_graph(SaveCluster, G, node):
                    save_AVG = []  # Initialize a list to save average coordinates
                    average_nodes = []  # Initialize a list to store average nodes
                    neighbor_lists = []  # Initialize a list to store neighbors of average nodes


                    for cluster_index, coordinates in enumerate(SaveCluster):
                        if len(coordinates)>1:


                            coorNode = eval(node)
                            # Calculate the average point
                            average_coordinate = [
                                (coorNode[0] + sum(point[0] for point in coordinates)) / (len(coordinates) + 1),
                                (coorNode[1] + sum(point[1] for point in coordinates)) / (len(coordinates) + 1),
                                (coorNode[2] + sum(point[2] for point in coordinates)) / (len(coordinates) + 1)
                            ]

                            # Print the average point
                            print("Average Point:", average_coordinate)
                            save_AVG.append(average_coordinate)

                            # Add the average coordinate to the graph
                            average_vertex = Vertex(average_coordinate)
                            currnode = Vertex(eval(node))
                            graph.add_node(str(average_vertex.point), pos=average_vertex.point)
                            graph.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
                            G.add_node(str(average_vertex.point), pos=average_vertex.point)
                            G.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
                            print(f'Cluster {cluster_index + 1} - Average Coordinate: {average_vertex.point}')
                            for t in range(len(coordinates)):
                                node_SaveCluster = Vertex(coordinates[t])
                                graph.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))


                            # Save the average node and its neighbors for later processing
                            average_nodes.append(average_vertex.point)
                            neighbor_lists.append(coordinates)

                            resulting_clusters_sav = coordinates
                            connected_edges = list(G.edges(node))
                            for edge in range(len(connected_edges)):
                                for nbn in range(len(resulting_clusters_sav)):
                                    if str(resulting_clusters_sav[nbn]) in connected_edges[edge]:
                                        G.remove_edge(*connected_edges[edge])
                                        graph.remove_edge(*connected_edges[edge])

                    return average_nodes, neighbor_lists



                neighbors1 = list(G.neighbors(node))
                resulting_clusters = cluster_neighbors(neighbors1)

                for s in range(len(resulting_clusters)):
                    if len(resulting_clusters[s]) >1:
                        average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
                        # print(average_nodes)
                        def process_new_node(node, visited, neighbors=None):
                            if node in visited:
                                return

                            visited.add(node)

                            degree = G.degree(node)
                            print(f'Degree of {node}: {degree}')

                            if degree > 3:
                                # Handle nodes with degree > 3
                                print(f'Processing new node with degree > 3: {node}')

                                # Add your logic here for processing new nodes with degree > 3
                                # You can create the average point, add it to the graph, and add edges

                                neighbors1 = list(G.neighbors(node))
                                resulting_clusters = cluster_neighbors(neighbors1)
                                # For example, you can use your existing code to create the average point and add it to the graph
                                average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
                                print(average_nodes)

                                # After adding the new node and edges, you can process the neighbors
                                if neighbors:
                                    for neighbor in neighbors:
                                        process_new_node(str(neighbor), visited)




            # Get the neighbors of the current node
            neighbors = list(G.neighbors(node))

            for neighbor in neighbors:
                process_node(neighbor, visited)


        for subgraph_node in dag.nodes:
            if eval(subgraph_node) in rootNodes:
                # Define your root node
                root_node = subgraph_node

        # Create a set to keep track of visited nodes
        visited_nodes = set()


        for ind, gnode  in enumerate(G1.nodes):
            if ind == 0:
                process_node(root_node, visited_nodes)
            else:
                if gnode not in visitNode:
                    # Start the traversal from the root node
                    process_node(gnode, visited_nodes)



    return graph



def graph_with_side_branches_group2(graph, rad):
    g = graph.copy() #one copy for reading the nodes from the perivious graph
    g1 = graph.copy() # another copy for adding the new branches


    avglen = []
    avgratiolendiam = []
    new_branches_lentodiam = []
    for u, v, o in g.edges(data="object"):

        if g[u][v]["RatioLendiameter"] <= 5:
             avgratiolendiam.append(g[u][v]["RatioLendiameter"])
             avglen.append(g[u][v]["length"])

    avg_ratioLenDiam = statistics.mean(avgratiolendiam)
    avg_ratioLenDiam = math.ceil(avg_ratioLenDiam)
    avg_len = statistics.mean(avglen)
    avg_len = math.ceil(avg_len)


    for  u, v, o in g.edges(data="object"):
        if g[u][v]["RatioLendiameter"] > 5:

                direction_vector = np.array(ast.literal_eval(v)) - np.array(ast.literal_eval(u))
                vector = direction_vector

            # for u, v, o in g.edges(data="object"):
                allpoints = []
                p = {}
                allpoints.append(u)
                allpoints.append(o.pixels)
                allpoints.append(v)
                flattened_list = [item if isinstance(item, str) else item[0] for sublist in allpoints for item in (sublist if isinstance(sublist, list) else [sublist])]

                branches = [(np.array(ast.literal_eval(u)), np.array(ast.literal_eval(v))),]

                g1.remove_edge(u, v)
                # g.remove_node(u)
                # g.remove_node(v)


                newlen = (g[u][v]["length"] / avg_len)
                # newlen = math.ceil(newlen) # round the float number to big
                newlen = math.floor(newlen) - 2 # round the float number to small
                divisions = newlen

                # new_branches = []
                for start_point, end_point in branches:
                    length = np.linalg.norm(end_point - start_point)
                    # direction_vector = (end_point - start_point) / length
                    # division_lengths = np.linspace(0, length, divisions + 1)
                    step = (end_point - start_point) / (divisions + 1)
                    new_nodes = [list(start_point)]

                    # for i in range(1, len(division_lengths)):
                    #     new_start_point = list(start_point + division_lengths[i-1] * direction_vector)
                    #     new_end_point = list(start_point + division_lengths[i] * direction_vector)
                    #     new_branches.append((new_start_point, new_end_point))
                    for i in range(1, divisions + 1):
                        new_node = start_point + i * step
                        new_nodes.append(list(new_node))

                    # Add the end node
                    new_nodes.append(list(end_point))
                    new_branches = []

                    for i in range(len(new_nodes) - 1):
                        segment = (new_nodes[i], new_nodes[i+1])
                        new_branches.append(segment)



                # if not all(component >= 0 for component in vector[:2]):
                #     new_branches = new_branches
                #
                # elif all(component >= 0 for component in vector[:2]):
                new_branches.reverse()
                new_branches = [tuple(reversed(item)) for item in new_branches]

                for w in range(len(new_branches)):
                    add_more_newbranches = []
                    if len(new_branches) > 0:
                        if not str((new_branches[w][0])) in g1.nodes() or not str((new_branches[w][1])) in g1.nodes():
                            if not str((new_branches[w][0])) in g1.nodes():
                                startpoint = Vertex(((new_branches[w][0])))
                                endpoint = Vertex(((new_branches[w][1])))
                                g1.add_node(str(startpoint.point), pos=startpoint.point)
                                g1.add_node(str(endpoint.point), pos=endpoint.point)
                                g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                            elif not str((new_branches[w][1])) in g1.nodes():
                                startpoint = Vertex(((new_branches[w][1])))
                                endpoint = Vertex(((new_branches[w][0])))
                                g1.add_node(str(startpoint.point), pos=startpoint.point)
                                g1.add_node(str(endpoint.point), pos=endpoint.point)
                                g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))


                                new_branch_start = startpoint.point
                                main_endpoint = endpoint.point

                                g1[str(new_branch_start)][str(main_endpoint)]['length'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))
                                g1[str(new_branch_start)][str(main_endpoint)]['radius'] = (g[u][v]['radius'])
                                g1[str(new_branch_start)][str(main_endpoint)]['diameter'] = 2*(g[u][v]['radius'])
                                g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start)) /  2*(g[u][v]['radius'])
                                # g1[str(new_branch_start)][str(main_endpoint)]['weight'] = -1
                                # new_branches_lentodiam.append(g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'])



                                if str(new_branch_start) not in rad:
                                    rad[str(new_branch_start)] = (g[u][v]['radius'])

                                if str(main_endpoint) not in rad:
                                    rad[str(main_endpoint)] = (g[u][v]['radius'])



                                # direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                # vector = direction_vector
                                # if not all(component <= 0 for component in vector):

                                if w % 2 == 0:


                                    direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                    normalized_direction = direction_vector / np.linalg.norm(direction_vector)

                                    # Calculate the angle in radians (40 degrees)
                                    angle_degrees = 40
                                    angle_radians = np.radians(angle_degrees)

                                    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])

                                    # Rotate the direction vector to get the new direction
                                    new_direction = np.dot(rotation_matrix, normalized_direction)


                                    # Calculate the length of the new branch (half of the main branch)
                                    new_branch_length = 0.5 * np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))

                                    # Calculate the new end node's coordinates
                                    new_end = np.array(new_branch_start) + new_branch_length * new_direction

                                    print(f"New end node coordinates: {new_end}")



                                    newpoint = Vertex((list(new_end)))
                                    g1.add_node(str(newpoint.point), pos=newpoint.point)
                                    g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))


                                    newpoint_end = newpoint.point
                                    print(newpoint_end)
                                    g1[str(new_branch_start)][str(newpoint_end)]['radius'] = 0.4 * (g[u][v]['radius'])
                                    g1[str(new_branch_start)][str(newpoint_end)]['length'] = new_branch_length
                                    g1[str(new_branch_start)][str(newpoint_end)]['diameter'] = 2* (0.4 * (g[u][v]['radius']))
                                    g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'] = new_branch_length /  2* (0.4 * (g[u][v]['radius']))
                                    # g1[str(startpoint.point)][str(newpoint.point)]['weight'] = -1
                                    rad[str(newpoint.point)] = (0.4 * (g[u][v]['radius']))
                                    new_branches_lentodiam.append(g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'])


                                    # g1[startpoint.point][newpoint.point]['length'] = half_main_branch_length

                                    # Print the new branch coordinates
                                    print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                                    # network_plot_3D(g, angle=0)
                                else:
                                    direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                    normalized_direction = direction_vector / np.linalg.norm(direction_vector)

                                    # Calculate the angle in radians (40 degrees)
                                    angle_degrees = -40
                                    angle_radians = np.radians(angle_degrees)

                                    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                    [0, 1, 0],
                                    [-np.sin(angle_radians), 0, np.cos(angle_radians)]])

                                    # Rotate the direction vector to get the new direction
                                    new_direction = np.dot(rotation_matrix, normalized_direction)


                                    # Calculate the length of the new branch (half of the main branch)
                                    new_branch_length = 0.5 * np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))

                                    # Calculate the new end node's coordinates
                                    new_end = np.array(new_branch_start) + new_branch_length * new_direction

                                    # print(f"New end node coordinates: {new_end}")



                                    newpoint = Vertex((list(new_end)))
                                    g1.add_node(str(newpoint.point), pos=newpoint.point)
                                    g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))

                                    newpoint_end = newpoint.point
                                    print(newpoint_end)
                                    g1[str(new_branch_start)][str(newpoint_end)]['radius'] = 0.4 * (g[u][v]['radius'])
                                    g1[str(new_branch_start)][str(newpoint_end)]['length'] = new_branch_length
                                    g1[str(new_branch_start)][str(newpoint_end)]['diameter'] = 2*(0.4 * (g[u][v]['radius']))
                                    g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'] = new_branch_length /  2* (0.4 * (g[u][v]['radius']))
                                    # g1[str(startpoint.point)][str(newpoint.point)]['weight'] = -1
                                    rad[str(newpoint.point)] = (0.4 * (g[u][v]['radius']))
                                    new_branches_lentodiam.append(g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'])

                                    # g1[startpoint.point][newpoint.point]['length'] = half_main_branch_length

                                    # Print the new branch coordinates
                                    print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                                    # network_plot_3D(g, angle=0)

                        elif str((new_branches[w][0])) in g1.nodes() and str((new_branches[w][1])) in g1.nodes():
                             new_branch_start = Vertex(new_branches[w][0])
                             main_endpoint = Vertex(new_branches[w][1])
                             g1.add_node(str(new_branch_start.point), pos=new_branch_start.point)
                             g1.add_node(str(main_endpoint.point), pos=main_endpoint.point)
                             g1.add_edge(str(new_branch_start.point), str(main_endpoint.point), object=Edge(new_branch_start, main_endpoint))


                             g1[str(new_branch_start)][str(main_endpoint)]['length'] = np.linalg.norm(np.array(main_endpoint.point) - np.array(new_branch_start.point))
                             g1[str(new_branch_start)][str(main_endpoint)]['radius'] = (g[u][v]['radius'])
                             g1[str(new_branch_start)][str(main_endpoint)]['diameter'] = 2 * (g[u][v]['radius'])
                             g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'] = np.linalg.norm(np.array(main_endpoint.point) - np.array(new_branch_start.point)) / 2 * (g[u][v]['radius'])
                             # g1[str(new_branch_start)][str(main_endpoint)]['weight'] = -1
                             # new_branches_lentodiam.append(g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'])

    avg_new_branches_lentodiam = statistics.mean(new_branches_lentodiam)  
    print(avg_new_branches_lentodiam)




    return g1, rad

def ratiolendiam_side_branches_group2(graph, rad):
    g = graph.copy()
    g1 = graph.copy()

    avglen = []
    avgratiolendiam = []
    new_branches_lentodiam = []
    new_branches_ratiodiam = []

    for u, v, o in g.edges(data="object"):
        if g[u][v]["RatioLendiameter"] <= 5:
            avgratiolendiam.append(g[u][v]["RatioLendiameter"])
            avglen.append(g[u][v]["length"])

    avg_ratioLenDiam = math.ceil(statistics.mean(avgratiolendiam))
    avg_len = math.ceil(statistics.mean(avglen))

    for u, v, o in g.edges(data="object"):
        if g[u][v]["RatioLendiameter"] > 5:
            direction_vector = np.array(ast.literal_eval(v)) - np.array(ast.literal_eval(u))

            allpoints = [u, o.pixels, v]
            flattened_list = [item if isinstance(item, str) else item[0] for sublist in allpoints for item in (sublist if isinstance(sublist, list) else [sublist])]

            branches = [(np.array(ast.literal_eval(u)), np.array(ast.literal_eval(v))),]

            g1.remove_edge(u, v)

            newlen = (g[u][v]["length"] / avg_len)
            newlen = math.floor(newlen) 
            divisions = newlen

            for start_point, end_point in branches:
                step = (end_point - start_point) / (divisions + 1)
                new_nodes = [list(start_point)]

                for i in range(1, divisions + 1):
                    new_node = start_point + i * step
                    new_nodes.append(list(new_node))

                new_nodes.append(list(end_point))
                new_branches = []

                for i in range(len(new_nodes) - 1):
                    segment = (new_nodes[i], new_nodes[i+1])
                    new_branches.append(segment)

                new_branches.reverse()
                new_branches = [tuple(reversed(item)) for item in new_branches]

                for w in range(len(new_branches)):
                    if len(new_branches) > 0:
                        if str((new_branches[w][0])) not in g1.nodes() or str((new_branches[w][1])) not in g1.nodes():
                            if not str((new_branches[w][0])) in g1.nodes():
                                startpoint = Vertex(((new_branches[w][0])))
                                endpoint = Vertex(((new_branches[w][1])))
                                g1.add_node(str(startpoint.point), pos=startpoint.point)
                                g1.add_node(str(endpoint.point), pos=endpoint.point)
                                g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                            elif not str((new_branches[w][1])) in g1.nodes():
                                startpoint = Vertex(((new_branches[w][1])))
                                endpoint = Vertex(((new_branches[w][0])))
                                g1.add_node(str(startpoint.point), pos=startpoint.point)
                                g1.add_node(str(endpoint.point), pos=endpoint.point)
                                g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))

                                new_branch_start = startpoint.point
                                main_endpoint = endpoint.point

                                g1[str(new_branch_start)][str(main_endpoint)]['length'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))
                                g1[str(new_branch_start)][str(main_endpoint)]['radius'] = (g[u][v]['radius'])
                                g1[str(new_branch_start)][str(main_endpoint)]['diameter'] = 2 * (g[u][v]['radius'])
                                g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start)) /  (2 * g[u][v]['radius'])

                                if str(new_branch_start) not in rad:
                                    rad[tuple(new_branch_start)] = (g[u][v]['radius'])

                                if str(main_endpoint) not in rad:
                                    rad[tuple(main_endpoint)] = (g[u][v]['radius'])

                                if w % 2 == 0:
                                    rotate_branch(g1, new_branch_start, main_endpoint, 40, rad, new_branches_lentodiam, new_branches_ratiodiam)
                                else:
                                    rotate_branch(g1, new_branch_start, main_endpoint, -40, rad, new_branches_lentodiam, new_branches_ratiodiam)

    avg_new_branches_lentodiam = statistics.mean(new_branches_lentodiam)
    std_dev_new_branches_lentodiam = statistics.stdev(new_branches_lentodiam)  # Added this line
    avg_new_branches_ratiodiam = statistics.mean(new_branches_ratiodiam)
    std_dev_new_branches_ratiodiam = statistics.stdev(new_branches_ratiodiam) 
    print(avg_new_branches_lentodiam)
    print(std_dev_new_branches_lentodiam)
    print(avg_new_branches_ratiodiam)
    print(std_dev_new_branches_ratiodiam)

    return g1, rad

def rotate_branch(graph, start, end, angle_degrees, rad, new_branches_lentodiam, new_branches_ratiodiam):
    direction_vector = np.array(end) - np.array(start)
    normalized_direction = direction_vector / np.linalg.norm(direction_vector)

    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                [0, 1, 0],
                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
    new_direction = np.dot(rotation_matrix, normalized_direction)

    new_branch_length = 0.5 * np.linalg.norm(np.array(end) - np.array(start))
    new_end = np.array(start) + new_branch_length * new_direction

    newpoint = Vertex(list(new_end))
    graph.add_node(str(newpoint.point), pos=newpoint.point)
    graph.add_edge(str(start), str(newpoint.point), object=Edge(start, newpoint))

    newpoint_end = newpoint.point
    graph[str(start)][str(newpoint_end)]['radius'] = 0.4 * (graph[str(start)][str(end)]['radius'])
    graph[str(start)][str(newpoint_end)]['length'] = new_branch_length
    graph[str(start)][str(newpoint_end)]['diameter'] = 2 * (0.4 * (graph[str(start)][str(end)]['radius']))
    graph[str(start)][str(newpoint_end)]['RatioLendiameter'] = new_branch_length / (2 * (0.4 * (graph[str(start)][str(end)]['radius'])))
    graph[str(start)][str(newpoint_end)]['ratioDiam'] =  graph[str(start)][str(newpoint_end)]['diameter'] / (graph[str(start)][str(end)]['diameter'])
    rad[str(newpoint.point)] = (0.4 * (graph[str(start)][str(end)]['radius']))
    
    new_branches_lentodiam.append(graph[str(start)][str(newpoint_end)]['RatioLendiameter'])
    new_branches_ratiodiam.append(graph[str(start)][str(newpoint_end)]['ratioDiam'])

def remove_small_subgraphs(graph, min_size):
    components = list(nx.weakly_connected_components(graph))

    for component in components:
        if len(component) < min_size:
            graph.remove_nodes_from(component)




#######################RUN#######################################################

# df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii_Artery.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii2_Artery.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Joyce/Vessel_Quantification/test_twobranches_15814w.csv', header=None)
df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii_Artery.csv', header=None) # the main data for 15814w
# df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii_Artery_translation.csv', header=None) # the translation data for 15814w
# df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii_Artery_ITK_translation.csv', header=None) # the translation data for 15814w


# df1 = pd.read_csv('/home/arah607/Desktop/left_test_15814W.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/test_15814w_node.csv', header=None)  #### example for latter for finding side branches
# df1 = pd.read_csv('/home/arah607/Desktop/test_15814w_node.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/15814w/15814w_coor_filterradii_Artery (copy).csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/outputGraph/coordinatesRUL.csv', header=None)


# df1 = pd.read_csv('/home/arah607/Desktop/16032X/16032X_ArteryLungVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16032X/16032X_ArteryLungVesselParticles_radiifilter15_translate.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16311B/16311B_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16311B/16311B_ArteryVesselParticles_radiifilter15_translation.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16315J/16315J_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16315J/16315J_ArteryVesselParticles_radiifilter15_translation.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16617Z/16617Z_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/16617Z/16617Z_ArteryVesselParticles_radiifilter15_translation.csv', header=None)

# df1 = pd.read_csv('/home/arah607/Desktop/17257A/17257A_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/17257A/17257A_ArteryVesselParticles_radiifilter15_translation.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/17275C/17275C_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/17929X/17929X_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/18347G/18347G_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/18615F/18615F_ArteryVesselParticles_radiifilter15.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/19020F/19020F_ArteryVesselParticles_radiifilter15.csv', header=None)


# df1 = pd.read_csv('/home/arah607/Desktop/test_15814W.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Desktop/smalltest_15814W.csv', header=None)


df1 = df1.sort_values(0) # sort decending csv file, read main artery
# df.sort_values(df.columns[0, 1, 2], axis=0, inplace=True) # sort from high to low


# Translation values, which found from image information in ITK (image processing tool)
x_origin = -158.7
y_origin = -360.2
z_origin = -295
z_spacing = 0.5  # Adjust this value according to your requirement
z_depth = 493


coor = []
radii={}
for index, row in df1.iterrows():
    coor.append([row[0], row[1], row[2]])
    radii[str([row[0], row[1], row[2]])] = row[3]
    # Translate the coordinates
    # translated_x = round(row[0] - x_origin, 3)
    # translated_y = round(row[1] - y_origin, 3)
    # translated_z = round(row[2] + (-1*(z_origin + z_spacing * z_depth)), 3)
    #
    # coor.append([translated_x, translated_y, translated_z])
    #
    # # Update the radii dictionary
    # radii[str([translated_x, translated_y, translated_z])] = row[3]




value = radii.values()
scale = list(value)
scale = np.array(scale)
sigma0 = 1/np.sqrt(2.)/2.
# spacing = np.array([0.625, 0.625, 0.625])
spacing = np.array([0.6289, 0.6289, 0.5])
selfspacing=np.prod(spacing)**(1/3.0)
sigmap = 1/np.sqrt(2.)/2.

# for i in range(len(scale)):
# scale = np.array([0.1, 0.2, 0.3])
mask = scale < (2. / np.sqrt(2) * sigma0)
rad = np.zeros(mask.shape)
rad[mask] = np.sqrt(2.) * (np.sqrt((scale[mask] * selfspacing) ** 2.0 + (
            sigma0 * selfspacing) ** 2.0) - 0.5 * sigma0 * selfspacing)
rad[~mask] = np.sqrt(2.) * (np.sqrt((scale[~mask] * selfspacing) ** 2.0 + (
            sigmap * selfspacing) ** 2.0) - 0.5 * sigmap * selfspacing)

new_dic = {key: rad[i] for i, key in enumerate(radii.keys())}
####################################### read the COPDGene_Phase1_SM_NS_25OCT21.txt file ####################################
import csv
#
# # Read the text file
# with open('/eresearch/lung/arah607/COPDgene/COPDGene_Phase1_SM_NS_25OCT21.txt', 'r') as text_file:
#     lines = text_file.readlines()
#
# # Prepare the data for CSV
# data = [line.strip().split('\t') for line in lines]


# # Write the data to a CSV file
# with open('output.csv', 'w', newline='') as csv_file:
#     writer = csv.writer(csv_file, delimiter='\t')
#     # writer.writerow(header)  # Write the header
#     writer.writerows(data)  # Write the data


#
# # Open the CSV file
# with open('output.csv', 'r') as csv_file:
#     reader = csv.reader(csv_file)
#
#     # Convert rows to array
#     data_array = [row for row in reader]

#
# row_index = 1  # Index of the row
# column_header = 'sid'  # Header of the column
#
# # Find the column index based on the header
# header_row = data_array[0]
# column_index = header_row.index(column_header)
#
# # Find the value in the matrix
# value = data_array[row_index][column_index]
# print(value)


############################################# read CIP vtk file ##############################################################

# Connceted_Tree('tree_CIP_right_13feb.csv', 'connected_tree.csv')

H = nx.Graph()
# combined_graph = connected_component_to_graph(coor,H)
# coor.sort(key=lambda x: x[0]) #sort coor with trunk coordinates
# add the all end points from pulmonary trunk graph to coordinates of the inside lung
G = buildTree_tree(coor, start=None)
# network_plot_3D(G, angle=0)
# mergedGraph = mergeEdges(G)
# df1 = pd.read_csv('/home/arah607/Joyce/Vessel_Quantification/test_twobranches_15814w.csv', header=None)
# df1 = df1.sort_values(0) # sort decending csv file, read main artery
# # df.sort_values(df.columns[0, 1, 2], axis=0, inplace=True) # sort from high to low
# coor = []
# radii={}
# for index, row in df1.iterrows():
#     coor.append([row[0], row[1], row[2]])
#     radii[str([row[0], row[1], row[2]])] = row[3]
Connected_G = coneccted_tree(G, new_dic)
# network_plot_3D(Connected_G, angle=0)
mergedGraph = mergeEdges(Connected_G)
# network_plot_3D(mergedGraph, angle=0)
trimmedGraph = removeSmallEdge(mergedGraph) # I should remove this line
remergedGraph = mergeEdges(trimmedGraph) # remove this line
# network_plot_3D(remergedGraph, angle=0)
connected_graph, separate_edges = remove_small_seprate_branches(remergedGraph)
# network_plot_3D(connected_graph, angle=0)
RmergedGraph = mergeEdges(connected_graph)
# network_plot_3D(RmergedGraph, angle=0)


Combined_graph = avgBranchRadius(RmergedGraph, new_dic)# Python implementation for

Final_graph_without_side, RootBranches = Final_graph(Combined_graph, new_dic)

writeGraph(Final_graph_without_side)
# network_plot_3D(Connect_graphs_mainArtery, angle=0)
graph_withsidebranches_group1 = graph_with_side_branches_group1(Final_graph_without_side, separate_edges, Connected_G)
# network_plot_3D(graph_withsidebranches_group1, angle=0)

mergedGraph1 = mergeEdges(graph_withsidebranches_group1)
# network_plot_3D(mergedGraph1, angle=0)
trimmedGraph1 = removeSmallEdge(mergedGraph1) # I should remove this line
remergedGraph1 = mergeEdges(trimmedGraph1) # remove this line
# network_plot_3D(remergedGraph, angle=0)
connected_graph1, separate_edges = remove_small_seprate_branches(remergedGraph1)
# network_plot_3D(connected_graph, angle=0)Final_graph_without_side
RmergedGraph1 = mergeEdges(connected_graph1)
Combined_graph1 = avgBranchRadius(RmergedGraph1, new_dic)# Python implementation for
# network_plot_3D(Combined_graph1, angle=0)

Final_graph_sidebranch_group1, RootBranches1 = Final_graph(Combined_graph1, new_dic)
# network_plot_3D(Final_graph_sidebranch_group1, angle=0)
writeGraph(Final_graph_sidebranch_group1)

# RootBranches1 = ['[-44.512, -131.635, -158.047]', '[-40.363, -131.414, -159.837', '[-55.232, -115.11, -164.441]', '[-50.312, -132.86, -173.283]', '[-57.82, -131.561, -185.691]', '[-49.644, -152.625, -176.111]', '[-46.189, -145.051, -189.359]', '[-44.979, -123.205, -204.194]', '[-51.512, -127.743, -205.413]', '[-46.757, -128.623, -205.363]', '[-56.777, -137.471, -211.137]', '[-51.533, -144.945, -209.056]', '[-61.389, -132.681, -232.511]',  '[-54.164, -133.158, -222.577]', '[32.455, -141.307, -164.76]', '[27.641, -125.905, -170.656]', '[38.082, -142.384, -170.972]', '[37.996, -123.043, -179.82]', '[35.161, -116.811, -183.591]', '[46.558, -136.565, -199.731]', '[39.585, -117.228, -190.413]', '[46.226, -134.455, -205.874]', '[52.238, -147.65, -249.847]', '[43.326, -121.745, -220.907]', '[40.701, -120.308, -233.437]', ]
# Final_graph_sidebranch_group1_removeDeg_bigger3 = Remove_Degree_BiggerThree(Final_graph_sidebranch_group1, RootBranches1)

# ###############################Find weakly connected components (remove small subgraphs in main graph)###############################
wccs = list(nx.weakly_connected_components(Final_graph_sidebranch_group1))

if len(wccs) >=1:
    # Set a threshold for the minimum size of a connected component
    min_component_size = 6  # Set your desired threshold

    # Identify components to remove
    components_to_remove = [component for component in wccs if len(component) < min_component_size]

    # Remove identified components
    for component in components_to_remove:
        for node in component:
            Final_graph_sidebranch_group1.remove_node(node)




############remove all nodes with degree greater than three ##################
import networkx as nx


# Identify nodes with degree greater than 3
nodes_to_reduce = [node for node in Final_graph_sidebranch_group1.nodes() if Final_graph_sidebranch_group1.degree(node) > 3]

# For each identified node, choose the outgoing edge with the smallest length, and remove it along with the connected node
for node in nodes_to_reduce:
    print(f"Node {node} has degree {G.degree(node)}")

    # Get information about outgoing edges (outlets)
    outgoing_edges = list(Final_graph_sidebranch_group1.out_edges(node, data=True))

    # Choose the edge with the smallest length
    edge_to_remove = min(outgoing_edges, key=lambda x: x[2].get('length', float('inf')))

    # Print information about the chosen edge
    print(f"  Removing edge: {edge_to_remove}")

    # Remove the chosen edge and the connected node
    Final_graph_sidebranch_group1.remove_node(edge_to_remove[1])


# graph_withsidebranches_group2, new_dic_new = graph_with_side_branches_group2(Final_graph_sidebranch_group1, new_dic)
# # RmergedGraph2 = mergeEdges(graph_withsidebranches_group2)
# # Combined_graph2 = avgBranchRadius_group2(graph_withsidebranches_group2, new_dic_new)
# Final_graph_sidebranch_group2, RootBranches2 = Final_graph_Group2(graph_withsidebranches_group2, new_dic_new)
# # network_plot_3D(Final_graph_sidebranch_group2, angle=0)
# # writeGraph(Final_graph_sidebranch_group2)
# 




# subgraphs = All_subgraphs(Final_graph_sidebranch_group1)
subgraphs = get_directed_subgraphs(Final_graph_sidebranch_group1, RootBranches1)

# connect graphs as much as possible
connect_graphs = connect_separate_GraphTrees(subgraphs, Final_graph_sidebranch_group1, new_dic)
writeGraph(connect_graphs)


################################### make reduction radius value from top to down of the graph ######################################
# # RootBranches1 = ['[-40.363, -131.414, -159.837]', '[-55.232, -115.11, -164.441]', '[-50.312, -132.86, -173.283]', '[-57.82, -131.561, -185.691]', '[-49.644, -152.625, -176.111]', '[-46.189, -145.051, -189.359]', '[-44.979, -123.205, -204.194]', '[-46.757, -128.623, -205.363]', '[-56.777, -137.471, -211.137]', '[-51.533, -144.945, -209.056]', '[-61.389, -132.681, -232.511]',  '[-54.164, -133.158, -222.577]', '[32.455, -141.307, -164.76]', '[27.641, -125.905, -170.656]', '[38.082, -142.384, -170.972]', '[37.996, -123.043, -179.82]', '[35.161, -116.811, -183.591]', '[46.558, -136.565, -199.731]', '[39.585, -117.228, -190.413]', '[46.226, -134.455, -205.874]', '[43.326, -121.745, -220.907]', '[40.701, -120.308, -233.437]']
# RootBranches1 = ['[115.688, 151.098, 223.255]', '[106.877, 135.558, 207.86]', '[101.156, 151.974, 210.498]', '[109.155, 156.066, 231.629]', '[109.309, 156.183, 228.322]', '[106.553, 136.98, 214.371]',
#                 ' [99.261, 165.985, 199.648]', '[110.004, 160.894, 197.801]', '[110.795, 155.535, 174.073]', '[102.16, 128.993, 188.485]', '[97.118, 135.078, 189.189]', '[100.772, 180.297, 177.039]',
#                  '[83.338, 170.992, 176.09]', '[91.715, 175.903, 173.961]', '[92.854, 180.133, 171.921]', '[88.823, 176.549, 170.918]', '[87.093, 191.62, 162.76]', '[103.493, 185.916, 180.063]',
#                  '[192.28, 173.01, 221.915]', '[187.828, 154.502, 226.502]', '[190.026, 133.831, 222.865]', '[195.118, 149.226, 205.007]', '[203.225, 130.243, 195.49]', '[201.525, 140.234, 205.779]',
#                  '[182.652, 147.727, 230.454]', '[186.936, 127.648, 223.08]', '[198.647, 172.665, 196.372]', '[202.585, 179.915, 194.305]', '[200.261, 158.09, 192.112]', '[207.962, 162.603, 187.637]',
#                  '[216.065, 162.137, 184.842]', '[193.407, 152.968, 146.308]', '[211.344, 157.128, 188.578]'
#                  ]
#
# for rootN in RootBranches1:
#     if connect_graphs.has_node(str(rootN)):
# # rootN = '[145.470187956702, 139.918452882767, 196.563]'
#         bfs_tree = nx.bfs_tree(connect_graphs, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)
#
#         # Find the parent of each node
#         parent = {}
#         for node in bfs_tree.nodes():
#             parent_nodes = list(bfs_tree.predecessors(node))
#             if parent_nodes:
#                 parent[node] = parent_nodes[0]
#             else:
#                 parent[node] = None
#
#
#         for edge in bfs_tree.edges():
#             parent[edge[1]] = edge[0]
#
#
#         # Print the parents of each node
#         print("Parents of each node:", parent)
#
#         check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
#         node_list = list(bfs_tree.nodes())
#         edge_list = list(bfs_tree.edges())
#
#         check = []
#         ratiocheck = {}
#         Diamcheck = False
#         for node in bfs_tree.nodes():
#
#             node = str(node)
#             # node_index = node_list.index(node)
#             node_index = list(bfs_tree.nodes()).index(node)
#             if check_node[node_index] == False:
#
#                 if parent[node] != None:
#
#                         par = connect_graphs[parent[node]][node]["radius"]
#
#                         if len(list(connect_graphs.neighbors(node))) > 0:
#                             nnb = list(connect_graphs.neighbors(node))
#
#                             # if str(rootN) in nnb:
#                             #     nnb.remove(str(rootN))
#                             queue = []
#                             for j in range(len(nnb)):
#                                 nnb_index = list(bfs_tree.nodes()).index(nnb[j])
#                                 if check_node[nnb_index] == False:
#                                     if par!= 0:
#                                         # children = nnb
#                                         child = connect_graphs[node][nnb[j]]["radius"]
#                                         if child != 0:
#                                             ratioDiam = (child / par)
#
#                                             check_node[node_index] = True
#                                             check.append(node)
#                                             # node_index = node_list.index(nnb[j])
#
#                                             # check_node[nnb_index] = True
#                                             if ratioDiam >1:
#
#                                                 connect_graphs[node][nnb[j]]["radius"] = par - (0.2 * par)
#                                                 connect_graphs[node][nnb[j]]["diameter"] = 2*(par - (0.2 * par))
#
#                                                 par = connect_graphs[parent[node]][node]["diameter"]
#                                                 child = connect_graphs[node][nnb[j]]["diameter"]
#                                                 ratioDiam = (child / par)
#                                                 connect_graphs[node][nnb[j]]["ratioDiam"] = ratioDiam
#                                         else:
#                                             connect_graphs[node][nnb[j]]["radius"] = par - (0.2* par)
#                                             connect_graphs[node][nnb[j]]["diameter"] = 2*(par - (0.2 * par))
#                                             child = connect_graphs[node][nnb[j]]["radius"]
#                                             ratioDiam = (child / par)
#
#                                             check_node[node_index] = True
#                                             check.append(node)
#                                             # node_index = node_list.index(nnb[j])
#
#                                             # check_node[nnb_index] = True
#                                             if ratioDiam >1:
#
#                                                 connect_graphs[node][nnb[j]]["radius"] = par - (0.2 * par)
#                                                 connect_graphs[node][nnb[j]]["diameter"] = 2*(par - (0.2 * par))
#
#                                                 par = connect_graphs[parent[node]][node]["diameter"]
#                                                 child = connect_graphs[node][nnb[j]]["diameter"]
#                                                 ratioDiam = (child / par)
#                                                 connect_graphs[node][nnb[j]]["ratioDiam"] = ratioDiam


AVG_Ratiolendiam  = ratiolendiam_side_branches_group2(Final_graph_sidebranch_group1, new_dic)
Connectgraph_withsidebranches_group2, Connectnew_dic_new = graph_with_side_branches_group2(Final_graph_sidebranch_group1, new_dic)
Final_Connectgraph_sidebranch_group2, RootBranches2 = Final_graph_Group2(Connectgraph_withsidebranches_group2, Connectnew_dic_new)
writeGraph(Final_Connectgraph_sidebranch_group2)


Join_Graphs = join_separate_GraphTrees(subgraphs, connect_graphs, new_dic)
# Join_Graphs = join_subgraphs_by_distance(subgraphs)
# network_plot_3D(Join_Graphs, angle=0)
# Connect_graphs_mainArtery = joint_graphs_mainArtery(Join_Graphs, new_dic, RootBranches)


# network_plot_3D(Connect_graphs_mainArtery, angle=0)
writeGraph(Join_Graphs)

################################# new idea#####################################
####################15814w#########################
# trunkpoint = [145.470187956702, 139.918452882767, 196.563]
#
Roots = []
Rootpoint_rul_15814w = [[115.688, 151.098, 223.255], [106.877, 135.558, 207.86], [101.156, 151.974, 210.498], [109.155, 156.066, 231.629], [109.309, 156.183, 228.322], [106.553, 136.98, 214.371]]
Rootpoint_rml_15814w = [[99.261, 165.985, 199.648], [110.004, 160.894, 197.801]]
Rootpoint_rll_15814w = [[110.795, 155.535, 174.073], [102.16, 128.993, 188.485], [97.118, 135.078, 189.189], [100.772, 180.297, 177.039], [83.338, 170.992, 176.09], [91.715, 175.903, 173.961], [92.854, 180.133, 171.921], [88.823, 176.549, 170.918], [87.093, 191.62, 162.76], [103.493, 185.916, 180.063]]

Rootpoint_lul_15814w = [[192.28, 173.01, 221.915], [187.828, 154.502, 226.502], [190.026, 133.831, 222.865], [195.118, 149.226, 205.007], [203.225, 130.243, 195.49], [201.525, 140.234, 205.779], [182.652, 147.727, 230.454], [186.936, 127.648, 223.08]]
Rootpoint_lll_15814w=[[198.647, 172.665, 196.372], [202.585, 179.915, 194.305], [200.261, 158.09, 192.112], [207.962, 162.603, 187.637], [216.065, 162.137, 184.842], [193.407, 152.968, 146.308], [211.344, 157.128, 188.578]]
# Rootpoint_rul_15814w = [[111.855, 184.266, -65.871], [118.388, 179.298, -74.245], [109.253, 165.18, -83.129], [109.577, 163.758, -89.64], [103.856, 180.174, -87.002], [137, 190.264, -88.356]]
# # node_rul =  [137, 190.264, -88.356] # this is a node which find in the mesh in cmgui
# Rootpoint_rml_15814w = [[104.86, 157.193, -109.015], [99.818, 163.278, -108.311], [102.97815, 185.40675, -118.55625]]
# # node_rml = [102.97815, 185.40675, -118.55625]
# Rootpoint_rll_15814w = [[101.961, 194.185, -97.852], [112.704, 189.094, -99.699], [113.495, 183.735, -123.427], [94.415, 204.103, -123.539], [91.523, 204.749, -126.582], [95.554, 208.333, -125.579], [103.472, 208.497, -120.461], [106.193, 214.116, -117.437], [87.417, 221.238, -130.908], [127.131, 198.92, -94.571]]
# # node_rll = [127.131, 198.92, -94.571]
# Rootpoint_lul_15814w = [[190.528, 182.702, -70.998], [185.352, 175.927, -67.046], [189.636, 155.848, -74.42], [192.726, 162.031, -74.635], [204.225, 168.434, -91.721], [205.925, 158.443, -102.01], [187.146, 163.81, -63.287], [186.168, 160.1, -79.213]]
# # nodes_lul = [[187.146, 163.81, -63.287], [186.168, 160.1, -79.213]]
# Rootpoint_lll_15814w = [[194.98, 201.21, -75.585], [201.347, 200.865, -101.128], [205.285, 208.115, -103.195], [210.662, 190.803, -109.863], [218.765, 190.337, -112.658], [202.961, 186.29, -105.388], [214.044, 185.328, -108.922], [200.37, 182.89, -93.192]]
# # node_lll = [200.37, 182.89, -93.192]

Roots.append(Rootpoint_rul_15814w)
Roots.append(Rootpoint_rml_15814w)
Roots.append(Rootpoint_rll_15814w)
Roots.append(Rootpoint_lul_15814w)
Roots.append(Rootpoint_lll_15814w)
#####################16032X######################
# trunkpoint = [158.011, 153.247, 201.52]
#
# Roots = []
# Rootpoint_rul_16032X = [ [117.485, 159.781, 212.815], [118.37, 158.009, 208.217], [112.947, 120.281, 190.225], [111.738, 116.885, 186.768]]#[100.465, 166.052, 205.881], [119.912, 149.887, 210.248],  [119.538, 129.739, 190.413], [112.562, 140.589, 196.075], [116.484, 137.807, 199.046] , [102.316, 154.927, 188.107]
# Rootpoint_rml_16032X = [[115.825, 168.444, 171.25], [104.565, 164.999, 173.794]] #[97.01, 202.288, 176.795]
# Rootpoint_rll_16032X = [[95.484, 142.741, 164.826], [96.019, 163.969, 149.946], [90.341, 165.088, 147.481], [98.655, 177.39, 138.816], [91.117, 182.866, 134.183], [102.317, 180.576, 137.912]] #[93.857, 130.005, 150.807]
#
# Rootpoint_lul_16032X = [[232.948, 143.752, 157.855],[233.451, 131.166, 187.194],[220.67, 132.586, 192.822],[226.597, 165.352, 181.008], [222.097, 151.193, 207.912],[220.149, 152.997, 209.161]] # (wrong root [248.845, 187.605, 211.278],), [242.42, 118.377, 150.576],[226.028, 134.838, 188.594] ,[245.327, 121.023, 151.726],[241.48, 103.329, 180.898],[235.253, 107.305, 180.171],[221.162, 160.312, 204.956], [220.464, 158.091, 202.026]
# Rootpoint_lll_16032X=[[227.589, 177.754, 166.065],[219.152, 185.297, 129.34],[222.517, 180.674, 128.579],[224.604, 167.489, 143.699], [240.865, 161.989, 113.037]] #[223.279, 175.553, 113.132], new: [227.22, 177.586, 174.068], [227.47, 182.538, 123.87],[213.319, 155.313, 82.73],[236.411, 157.296, 122.172],[246.849, 170.735, 137.662]
#
# Roots.append(Rootpoint_rul_16032X)
# Roots.append(Rootpoint_rml_16032X)
# Roots.append(Rootpoint_rll_16032X)
# Roots.append(Rootpoint_lul_16032X)
# Roots.append(Rootpoint_lll_16032X)
#########################16311B##################
# trunkpoint = [158.011, 150.247, 191.52]
#
# Roots = []
# Rootpoint_rul_16311B = [[135.124, 139.084, 236.991], [134.036, 138.499, 217.131], [134.442, 134.617, 218.64], [128.442, 127.977, 205.714], [125.937, 138.777, 195.0]]
# Rootpoint_rml_16311B = [[118.795, 141.181, 185.052], [128.385, 151.649, 187.491], [123.297, 131.964, 181.314]]
# Rootpoint_rll_16311B = [[120.993, 149.864, 167.822], [122.209, 159.921, 168.713], [118.471, 161.4, 168.743], [124.352, 162.741, 167.796], [127.446, 147.143, 159.023], [125.001, 148.965, 166.862]]
# Rootpoint_lul_16311B = [[199.15, 131.33, 208.316], [205.535, 126.635, 210.187], [208.303, 141.442, 194.436], [211.667, 132.219, 190.962], [200.409, 175.358, 188.584], [205.569, 170.798, 191.01], [205.924, 164.648, 197.331], [197.595, 143.382, 214.597]]
# Rootpoint_lll_16311B= [[201.466, 174.051, 160.732], [205.839, 168.246, 164.62], [204.783, 156.824, 178.455]]
#
# Roots.append(Rootpoint_rul_16311B)
# Roots.append(Rootpoint_rml_16311B)
# Roots.append(Rootpoint_rll_16311B)
# Roots.append(Rootpoint_lul_16311B)
# Roots.append(Rootpoint_lll_16311B)
#########################16315J######################
# trunkpoint = [158.011, 150.247, 191.52]
#
# Roots = []
# Rootpoint_rul_16315 = [[115.815, 126.053, 201.037], [123.431, 123.979, 232.486], [118.072, 122.703, 214.587], [111.046, 108.972, 205.976]]
# Rootpoint_rml_16315 = [[108.138, 120.621, 174.151], [117.216, 151.813, 183.615]]
# Rootpoint_rll_16315 = [[109.925, 141.873, 160.713], [108.503, 102.882, 161.159], [106.829, 134.223, 161.498], [113.335, 131.131, 158.396]]
#
# Rootpoint_lul_16315 = [[191.971, 63.093, 219.46], [215.522, 140.387, 230.212], [202.98, 134.216, 218.181], [211.582, 152.968, 186.158], [218.913, 133.156, 185.855], [214.982, 141.393, 202.662], [210.375, 150.001, 200.712]]
# Rootpoint_lll_16315= [[212.655, 154.29, 160.526], [209.358, 138.664, 134.012], [226.014, 141.307, 152.247], [224.41, 148.332, 164.927]]
# Roots.append(Rootpoint_rul_16315)
# Roots.append(Rootpoint_rml_16315)
# Roots.append(Rootpoint_rll_16315)
# Roots.append(Rootpoint_lul_16315)
# Roots.append(Rootpoint_lll_16315)
##########################16617Z##############################
# trunkpoint = [153.011, 145.247, 208.52]
#
# Roots = []
# Rootpoint_rul_16617Z =[[107.638, 165.176, 207.05], [98.42, 139.159, 203.68], [97.268, 158.893, 227.29], [113.658, 154.803, 244.25], [87.953, 130.414, 197.24], [89.536, 118.852, 184.9], [91.579, 119.57, 193.62], [106.289, 136.967, 234.18], [107.023, 146.465, 252.69], [95.746, 135.542, 235.15], [98.186, 122.807, 236.3]]
# Rootpoint_rml_16617Z = [[97.472, 150.429, 175.69], [92.243, 166.441, 192.54], [88.917, 159.919, 178.729], [81.161, 158.671, 177.168]]
# Rootpoint_rll_16617Z = [ [80.522, 162.186, 157.683], [84.022, 166.119, 159.787], [91.166, 171.664, 174.741], [95.383, 167.947, 174.916], [100.703, 169.779, 175.627]]
# Rootpoint_lul_16617Z =  [[197.054, 179.503, 208.19], [202.515, 135.581, 202.01],[206.634, 140.463, 202.04],[204.443, 159.921, 210.9], [203.97, 133.606, 218.53],[192.153, 152.021, 235.12],[190.896, 140.603, 238.28], [191.304,166.694, 210.29], [204.742, 117.788, 218.12]]
# Rootpoint_lll_16617Z= [[207.39, 171.789, 170.544], [209.771, 168.355, 169.725],[205.23, 149.32, 178.831], [206.063, 156.631, 187.97]]
# Roots.append(Rootpoint_rul_16617Z)
# Roots.append(Rootpoint_rml_16617Z)
# Roots.append(Rootpoint_rll_16617Z)
# Roots.append(Rootpoint_lul_16617Z)
# Roots.append(Rootpoint_lll_16617Z)
##################################17257A#############################
# trunkpoint = [153.011, 161.247, 163.52]
#
# Roots = []
# Rootpoint_rul_17257A =[[125.788, 145.162, 181.949], [129.813, 142.131, 185.77], [131.633, 130.711, 180.23], [125.995, 126.33, 160.534]]
# Rootpoint_rml_17257A = [[115.078, 151.82, 151.434], [120.174, 135.212, 135.771]] #[122.883, 175.511, 158.604]
# Rootpoint_rll_17257A = [[95.828, 153.097, 117.877], [111.492, 163.312, 137.856], [118.172, 166.721, 135.08], [101.633, 158.81, 117.956], [92.827, 149.699, 101.355], [96.797, 148.205, 96.452]]
# Rootpoint_lul_17257A =  [[200.722, 165.562, 188.852], [199.467, 160.918, 182.474], [204.933, 146.842, 181.697], [205.539, 150.689, 180.319], [204.03, 170.733, 172.348], [215.733, 156.42, 153.597], [214.678, 166.819, 153.843], [227.478, 167.742, 151.919], [202.262, 177.518, 163.329], [203.714, 190.444, 163.259]]
# Rootpoint_lll_17257A= [[215.202, 183.213, 135.499], [229.142, 168.154, 112.185], [211.803, 184.987, 123.545], [195.642, 188.764, 147.817], [200.827, 196.026, 106.68]]
# Roots.append(Rootpoint_rul_17257A)
# Roots.append(Rootpoint_rml_17257A)
# Roots.append(Rootpoint_rll_17257A)
# Roots.append(Rootpoint_lul_17257A)
# Roots.append(Rootpoint_lll_17257A)
##################################17275C#############################
# trunkpoint = [14.825, -153.636, -94.129]
#
# Roots = []
# Rootpoint_rul_17275C =[[-19.755, -165.697, -86.695], [-28.594, -157.575, -74.038], [-23.837, -149.635, -68.298], [-6.675, -155.93, -62.459]]
# Rootpoint_rml_17275C = [[-25.54, -156.466, -107.001]]
# Rootpoint_rll_17275C = [[-32.449, -130.98, -118.372], [-21.497, -137.734, -127.559], [-19.445, -132.74, -132.068], [-29.665, -119.549, -121.365], [-45.411, -120.415, -139.796], [-33.586, -88.856, -155.35]]
# Rootpoint_lul_17275C =  [[62.691, -143.803, -108.774], [62.503, -148.373, -102.119], [51.32, -146.594, -73.299], [58.85, -159.845, -74.514], [57.26, -163.932, -92.381], [55.733, -157.962, -62.122], [54.229, -168.012, -74.103]]
# Rootpoint_lll_17275C= [[72.727, -134.699, -115.84], [68.085, -122.948, -111.907], [67.096, -115.192, -121.02], [68.764, -111.653, -143.718], [59.38, -109.426, -136.467]]
# Roots.append(Rootpoint_rul_17275C)
# Roots.append(Rootpoint_rml_17275C)
# Roots.append(Rootpoint_rll_17275C)
# Roots.append(Rootpoint_lul_17275C)
# Roots.append(Rootpoint_lll_17275C)
############################17929X##########################################
# trunkpoint = [-1.989, -149.753, -149.48]
#
# Roots = []
# Rootpoint_rul_17929X =[[-60.17, -164.414, -142.961], [-59.534, -152.431, -135.578], [-52.181, -168.447, -132.284], [-48.629, -175.024, -140.04], [-55.515, -184.37, -139.526], [-43.545, -165.322, -122.551], [-48.536, -157.749, -109.379]]
# Rootpoint_rml_17929X = [[-55.273, -168.63, -163.778], [-62.116, -156.787, -169.726], [-55.485, -145.618, -157.668], [-52.744, -147.602, -158.792], [-73.659, -162.546, -173.096]]
# Rootpoint_rll_17929X = [[-76.959, -144.16, -181.536], [-67.32, -143.227, -178.971], [-64.078, -137.073, -179.352], [-63.55, -130.264, -179.687]]
# Rootpoint_lul_17929X =  [[38.705, -134.006, -141.888], [55.686, -155.685, -160.111], [51.38, -151.189, -153.321], [36.409, -150.631, -127.541], [43.644, -167.023, -133.055]]
# Rootpoint_lll_17929X= [[50.104, -132.5, -173.981], [64.247, -138.504, -188.389], [58.599, -132.922, -186.963], [48.984, -118.67, -182.183], [45.673, -114.659, -207.438], [48.492, -115.897, -207.027]] #, [42.877, -115.654, -194.595]
# Roots.append(Rootpoint_rul_17929X)
# Roots.append(Rootpoint_rml_17929X)
# Roots.append(Rootpoint_rll_17929X)
# Roots.append(Rootpoint_lul_17929X)
# Roots.append(Rootpoint_lll_17929X)
################################19020F##########################
# trunkpoint = [-5.175, -129.636, -200.129]
#
# Roots = []
# Rootpoint_rul_19020F =[[-57.82, -131.561, -185.691], [-50.312, -132.86, -173.283], [-49.644, -152.625, -176.111], [-40.363, -131.414, -159.837]]
# Rootpoint_rml_19020F = [[-46.757, -128.623, -205.363], [-51.533, -144.945, -209.056]]
# Rootpoint_rll_19020F = [[-61.389, -132.681, -232.511], [-54.164, -133.158, -222.577]]
# Rootpoint_lul_19020F =  [[35.161, -116.811, -183.591], [38.082, -142.384, -170.972], [27.641, -125.905, -170.656]]
# Rootpoint_lll_19020F= [[46.558, -136.565, -199.731], [43.326, -121.745, -220.907], [40.701, -120.308, -233.437]]
# Roots.append(Rootpoint_rul_19020F)
# Roots.append(Rootpoint_rml_19020F)
# Roots.append(Rootpoint_rll_19020F)
# Roots.append(Rootpoint_lul_19020F)
# Roots.append(Rootpoint_lll_19020F)

################################18615F##########################
# trunkpoint = [-15.175, 2.364, -156.129]
#
# Roots = []
# Rootpoint_rul_18615F =[[-52.493, 1.392, -126.204], [-50.477, 3.203, -130.298], [-67.438, 18.821, -124.295], [-60.414, -18.521, -133.151]]
# Rootpoint_rml_18615F = [[-74.04, -21.4, -175.564], [-63.942, -8.31, -166.495], [-56.509, 11.419, -160.811], [-59.729, 25.278, -163.771]]
# Rootpoint_rll_18615F = [[-71.631, 9.446, -185.716], [-68.444, 15.417, -183.837], [-74.74, 25.89, -199.504]]
# Rootpoint_lul_18615F =  [[37.459, -5.987, -117.461], [31.518, 9.313, -121.06], [36.18, 35.776, -131.491], [48.342, 27.017, -144.196], [55.001, 2.361, -149.984], [67.279, -1.527, -148.645], [57.812, 1.298, -148.909]]
# Rootpoint_lll_18615F= [[44.354, 28.386, -156.395], [58.321, 28.675, -166.287], [68.308, 25.345, -185.994], [68.256, 40.895, -185.189], [59.95, 19.149, -198.729]]
# Roots.append(Rootpoint_rul_18615F)
# Roots.append(Rootpoint_rml_18615F)
# Roots.append(Rootpoint_rll_18615F)
# Roots.append(Rootpoint_lul_18615F)
# Roots.append(Rootpoint_lll_18615F)
################### Recalculate the radius value in each seprate graph #############

# Final_graph = Final_Finalgraph(Final_graph_sidebranch_group2, Roots)

########################################## connect trunk graph to the join-graph ############################################
# # Assuming euclidean_distance is defined like this
# def euclidean_distance(point1, point2):
#      return math.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))
#
# for v in range(len(Roots)):
#     node_mesh = []
#     for n in range(len(Roots[v])):
#         if str(Roots[v][n]) not in Join_Graphs.nodes():
#             node_mesh.append(Roots[v][n])
#     Root_points = Roots[v]
#     g = Join_Graphs.copy()
#     # g = Final_graph_sidebranch_group2.copy()
#     components = list(nx.weakly_connected_components(g))
#
#     # Create a dictionary to store component labels
#     component_labels = {}
#
#     # Assign labels to each component
#     for i, component in enumerate(components):
#         for node in component:
#             component_labels[node] = i + 1
#
#     # Print the component labels
#     for node, label in component_labels.items():
#         print(f"Node {node}: Component {label}")
#     unique_labels = set(component_labels.values())
#     label_array = np.array(list(unique_labels))
#     # Print the unique labels
#     print("Unique Labels:")
#     for label in unique_labels:
#         print(label)
#
#     selected_label = len(unique_labels)
#
#     label_root_store = {}
#     subgraph = nx.DiGraph()
#
#     for i in range(selected_label):
#         nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
#         edges_in_label = [edge for edge in g.edges() if
#                           component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]
#
#
#         # Convert your 'a' list to a list of lists
#         a = [[float(coord) for coord in point.strip('[]').split(', ')] for point in nodes_in_label]
#
#         # Convert 'Rootpoint_lul' to a similar format
#         Root_points = [[float(coord) for coord in point] for point in Root_points]
#         # Root_points = [ast.literal_eval(item) for item in Root_points]
#
#         # Initialize a list to store matching elements
#         matching_elements = []
#
#         # Iterate through 'Rootpoint_lul' and 'a' to find matches
#         for point in Root_points:
#             if point in a:
#                 matching_elements.append(point)
#
#         # Print the matching elements
#         print("Matching elements in 'Rootpoint_lul':")
#         for element in matching_elements:
#             print(element)
#
#
#         # coor_set = set(tuple(point) for point in Rootpoint_rul)
#         if len(matching_elements) > 0:
#             for node in nodes_in_label:
#                 # if tuple(eval(node)) in coor_set:
#                     subgraph.add_node(node)
#             # Iterate through the edges in Join_Graphs and add edges that connect nodes in the subgraph
#             for edge in edges_in_label:
#                 node1, node2 = edge
#                 if node1 in subgraph.nodes and node2 in subgraph.nodes:
#                     subgraph.add_edge(node1, node2)
#
#
#     ######################### label subgraph ####################
#
#     g1 = subgraph.copy()
#     components = list(nx.weakly_connected_components(g1))
#
#     # Create a dictionary to store component labels
#     component_labels = {}
#
#     # Assign labels to each component
#     for i, component in enumerate(components):
#         for node in component:
#             component_labels[node] = i + 1
#
#     # Print the component labels
#     for node, label in component_labels.items():
#         print(f"Node {node}: Component {label}")
#     unique_labels = set(component_labels.values())
#     label_array = np.array(list(unique_labels))
#     # Print the unique labels
#     print("Unique Labels:")
#     for label in unique_labels:
#         print(label)
#
#     selected_label = len(unique_labels)
#
#     closest_nodes=[]
#
#     for i in range(selected_label):
#         nodes_in_label1 = [node for node, label in component_labels.items() if label == label_array[i]]
#         edges_in_label1 = [edge for edge in g1.edges() if
#                           component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]
#
#
#
#         # Convert 'nodes_in_label' to a list of lists
#         nodes_in_label1 = [[float(coord) for coord in point.strip('[]').split(', ')] for point in nodes_in_label1]
#
#
#
#         if len(node_mesh) == 1:
#            if str(node_mesh[0]) not in Join_Graphs.nodes():
#                node_lobe = Vertex(node_mesh[0])
#                Join_Graphs.add_node(str(node_lobe.point), pos=node_lobe.point)
#            # for k in range(len(nodes_in_label1)):
#            matching_nodes = [node for node in Root_points if any(all(val in sublist for val in node) for sublist in nodes_in_label1)]
#            if len(matching_nodes) == 1:
#                 node_root = Vertex(matching_nodes[0])
#                 Join_Graphs.add_edge(str(node_lobe.point), str(node_root.point), object=Edge(node_lobe, node_root))
#                 Join_Graphs[str(node_lobe.point)][str(node_root.point)]['radius'] = 1
#                 Join_Graphs[str(node_lobe.point)][str(node_root.point)]['diameter'] = 2
#                 distance = euclidean_distance(tuple(node_lobe.point), tuple(node_root.point))
#                 Join_Graphs[str(node_lobe.point)][str(node_root.point)]['length'] = distance
#            else:
#                for l in range(len(matching_nodes)):
#                    #chose_closest_node_to_root  Function to calculate Euclidean distance between two points
#                     def euclidean_distance(point1, point2):
#                         return math.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))
#
#                     # Calculate distances and find the node with minimum distance
#                     Choose_root_node = min(matching_nodes, key=lambda node: euclidean_distance(node, node_lobe.point))
#                     min_distance_value = euclidean_distance(Choose_root_node, node_lobe.point)
#                     print("Node with minimum distance:", Choose_root_node)
#                     node_root = Vertex(Choose_root_node)
#                     Join_Graphs.add_edge(str(node_lobe.point), str(node_root.point), object=Edge(node_lobe, node_root))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['radius'] = 1
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['diameter'] = 2
#                     distance = euclidean_distance(tuple(node_lobe.point), tuple(node_root.point))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['length'] = distance
#                     break
#
#         # it means when we have two nodes in a lul mesh
#         elif len(node_mesh) == 2:
#             for t in range(len(node_mesh)):
#                 if str(node_mesh[t]) not in Join_Graphs.nodes():
#                     node_lobe = Vertex(node_mesh[t])
#                     Join_Graphs.add_node(str(node_lobe.point), pos=node_lobe.point)
#
#             #### cluster the Root_points in lul
#             from sklearn.cluster import KMeans
#             import numpy as np
#             save_cluster_root_nodes_lul = []
#
#             # List of 3D points
#             b = Root_points
#             if len(b) > 1:
#                 # Convert the list to a NumPy array
#                 b_array = np.array(b)
#                 # Create a KMeans clustering model with 2 clusters
#                 kmeans = KMeans(n_clusters=2, random_state=0)
#                 # Fit the model to the data
#                 kmeans.fit(b_array)
#                 # Get cluster labels
#                 labels = kmeans.labels_
#                 # Separate the points into two clusters based on the labels
#                 cluster_1 = [b[i] for i in range(len(b)) if labels[i] == 0]
#                 cluster_2 = [b[i] for i in range(len(b)) if labels[i] == 1]
#                 save_cluster_root_nodes_lul.append(cluster_1)
#                 save_cluster_root_nodes_lul.append(cluster_2)
#                 # Print the two clusters
#                 print("Cluster 1:")
#                 for point in cluster_1:
#                     print(point)
#                 print("\nCluster 2:")
#                 for point in cluster_2:
#                     print(point)
#             else:
#                 save_cluster_root_nodes_lul.append(b)
#
#             ################ assigne each cluster to each node_mesh############
#             matching_nodes = [node for node in Root_points if any(all(val in sublist for val in node) for sublist in nodes_in_label1)]
#             all_in_cluster_0 = all(node in save_cluster_root_nodes_lul[0] for node in matching_nodes)
#             all_in_cluster_1 = all(node in save_cluster_root_nodes_lul[1] for node in matching_nodes)
#             #chose_closest_node_to_root  Function to calculate Euclidean distance between two points
#             def euclidean_distance(point1, point2):
#                 return math.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))
#
#             # Function to find the closest node in node_mesh to a cluster
#             def find_closest_node_to_cluster(cluster, nodes):
#                 closest_node = min(nodes, key=lambda node: min(euclidean_distance(node, point) for point in cluster))
#                 return closest_node
#
#             # Find the closest nodes for each cluster
#             closest_node_to_cluster_0 = find_closest_node_to_cluster(save_cluster_root_nodes_lul[0], node_mesh)
#             closest_node_to_cluster_1 = find_closest_node_to_cluster(save_cluster_root_nodes_lul[1], node_mesh)
#
#             print("Closest node to save_cluster_root_nodes_lul[0]:", closest_node_to_cluster_0)
#             print("Closest node to save_cluster_root_nodes_lul[1]:", closest_node_to_cluster_1)
#             if all_in_cluster_0 and closest_node_to_cluster_0:
#                 if len(matching_nodes) == 1:
#                     node_lobe = closest_node_to_cluster_0
#                     node_lobe = Vertex(node_lobe)
#                     node_root = Vertex(matching_nodes[0])
#                     Join_Graphs.add_edge(str(node_lobe.point), str(node_root.point), object=Edge(node_lobe, node_root))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['radius'] = 1
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['diameter'] = 2
#                     distance = euclidean_distance(tuple(node_lobe.point), tuple(node_root.point))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['length'] = distance
#                 else:
#                     for l in range(len(matching_nodes)):
#                        #chose_closest_node_to_root  Function to calculate Euclidean distance between two points
#                         def euclidean_distance(point1, point2):
#                             return math.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))
#
#                         # Calculate distances and find the node with minimum distance
#                         Choose_root_node = min(matching_nodes, key=lambda node: euclidean_distance(node, closest_node_to_cluster_0))
#                         min_distance_value = euclidean_distance(Choose_root_node, closest_node_to_cluster_0)
#                         print("Node with minimum distance:", Choose_root_node)
#                         node_root = Vertex(Choose_root_node)
#                         closest_nodes_cluster_0 = Vertex(closest_node_to_cluster_0)
#                         Join_Graphs.add_edge(str(closest_nodes_cluster_0.point), str(node_root.point), object=Edge(closest_nodes_cluster_0, node_root))
#                         Join_Graphs[str(closest_nodes_cluster_0.point)][str(node_root.point)]['radius'] = 1
#                         Join_Graphs[str(closest_nodes_cluster_0.point)][str(node_root.point)]['diameter'] = 2
#                         distance = euclidean_distance(tuple(closest_nodes_cluster_0.point), tuple(node_root.point))
#                         Join_Graphs[str(closest_nodes_cluster_0.point)][str(node_root.point)]['length'] = distance
#                         break
#
#
#             elif all_in_cluster_1 and closest_node_to_cluster_1:
#                 if len(matching_nodes) == 1:
#                     node_lobe = closest_node_to_cluster_1
#                     node_lobe = Vertex(node_lobe)
#                     node_root = Vertex(matching_nodes[0])
#                     Join_Graphs.add_edge(str(node_lobe.point), str(node_root.point), object=Edge(node_lobe, node_root))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['radius'] = 1
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['diameter'] = 2
#                     distance = euclidean_distance(tuple(node_lobe.point), tuple(node_root.point))
#                     Join_Graphs[str(node_lobe.point)][str(node_root.point)]['length'] = distance
#                 else:
#                     for l in range(len(matching_nodes)):
#                        #chose_closest_node_to_root  Function to calculate Euclidean distance between two points
#                         def euclidean_distance(point1, point2):
#                             return math.sqrt(sum((a - b)**2 for a, b in zip(point1, point2)))
#
#                         # Calculate distances and find the node with minimum distance
#                         Choose_root_node = min(matching_nodes, key=lambda node: euclidean_distance(node, closest_node_to_cluster_1))
#                         min_distance_value = euclidean_distance(Choose_root_node, closest_node_to_cluster_1)
#                         print("Node with minimum distance:", Choose_root_node)
#                         node_root = Vertex(Choose_root_node)
#                         closest_nodes_cluster_1 = Vertex(closest_node_to_cluster_1)
#                         Join_Graphs.add_edge(str(closest_nodes_cluster_1.point), str(node_root.point), object=Edge(closest_nodes_cluster_1, node_root))
#                         Join_Graphs[str(closest_nodes_cluster_1.point)][str(node_root.point)]['radius'] = 1
#                         Join_Graphs[str(closest_nodes_cluster_1.point)][str(node_root.point)]['diameter'] = 2
#                         distance = euclidean_distance(tuple(closest_nodes_cluster_1.point), tuple(node_root.point))
#                         Join_Graphs[str(closest_nodes_cluster_1.point)][str(node_root.point)]['length'] = distance
#                         break
#
#
#
#
# trunkpoint = [149.011, 170.247, -146.48]
# trunkpoint1 = [149.011, 170.247, -136.48]
# trunkpoint3 = [149.011, 170.247, -126.48]
# trunkpoint4 = [149.011, 170.247, -116.48]
# trunkpoint5 = [149.011, 170.247, -106.48]
# trunkpoint6 = [149.011, 170.247, -96.48]
# trunkpoint7 = [137, 190.264, -88.356]
# trunkpoint8 = [127.131, 198.92, -94.571]
# trunkpoint9 = [119.429, 201.075, -105.86]
# trunkpoint10 = [110.244, 195.331, -114.281]
# trunkpoint11 = [102.97815, 185.40675, -118.55625]
# trunkpoint12 = [147.353, 147.562, -66.322]
# trunkpoint13 = [161.372, 152.916, -68.747]
# trunkpoint14 = [177.252, 159.703, -67.55]
# trunkpoint15 = [187.146, 163.81, -63.287]
# trunkpoint16 = [186.168, 160.1, -79.213]
# trunkpoint17 = [200.37, 182.89, -93.192]
#
# trunkpoint = Vertex(trunkpoint)
# trunkpoint1 = Vertex(trunkpoint1)
# trunkpoint3 = Vertex(trunkpoint3)
# trunkpoint4 = Vertex(trunkpoint4)
# trunkpoint5 = Vertex(trunkpoint5)
# trunkpoint6 = Vertex(trunkpoint6)
# trunkpoint7 = Vertex(trunkpoint7)
# trunkpoint8 = Vertex(trunkpoint8)
# trunkpoint9 = Vertex(trunkpoint9)
# trunkpoint10 = Vertex(trunkpoint10)
# trunkpoint11 = Vertex(trunkpoint11)
# trunkpoint12 = Vertex(trunkpoint12)
# trunkpoint13 = Vertex(trunkpoint13)
# trunkpoint14 = Vertex(trunkpoint14)
# trunkpoint15 = Vertex(trunkpoint15)
# trunkpoint16 = Vertex(trunkpoint16)
# trunkpoint17 = Vertex(trunkpoint17)
#
# Join_Graphs.add_node(str(trunkpoint.point), pos=trunkpoint.point)
# Join_Graphs.add_node(str(trunkpoint1.point), pos=trunkpoint1.point)
# Join_Graphs.add_node(str(trunkpoint3.point), pos=trunkpoint3.point)
# Join_Graphs.add_node(str(trunkpoint4.point), pos=trunkpoint4.point)
# Join_Graphs.add_node(str(trunkpoint5.point), pos=trunkpoint5.point)
# Join_Graphs.add_node(str(trunkpoint6.point), pos=trunkpoint6.point)
# Join_Graphs.add_node(str(trunkpoint7.point), pos=trunkpoint7.point)
# Join_Graphs.add_node(str(trunkpoint8.point), pos=trunkpoint8.point)
# Join_Graphs.add_node(str(trunkpoint9.point), pos=trunkpoint9.point)
# Join_Graphs.add_node(str(trunkpoint10.point), pos=trunkpoint10.point)
# Join_Graphs.add_node(str(trunkpoint11.point), pos=trunkpoint11.point)
# Join_Graphs.add_node(str(trunkpoint12.point), pos=trunkpoint12.point)
# Join_Graphs.add_node(str(trunkpoint13.point), pos=trunkpoint13.point)
# Join_Graphs.add_node(str(trunkpoint14.point), pos=trunkpoint14.point)
# Join_Graphs.add_node(str(trunkpoint15.point), pos=trunkpoint15.point)
# Join_Graphs.add_node(str(trunkpoint16.point), pos=trunkpoint16.point)
# Join_Graphs.add_node(str(trunkpoint17.point), pos=trunkpoint17.point)
#
#
# Join_Graphs.add_edge(str(trunkpoint.point), str(trunkpoint1.point), object=Edge(trunkpoint, trunkpoint1))
# Join_Graphs.add_edge(str(trunkpoint1.point), str(trunkpoint3.point), object=Edge(trunkpoint1, trunkpoint3))
# Join_Graphs.add_edge(str(trunkpoint3.point), str(trunkpoint4.point), object=Edge(trunkpoint3, trunkpoint4))
# Join_Graphs.add_edge(str(trunkpoint4.point), str(trunkpoint5.point), object=Edge(trunkpoint4, trunkpoint5))
# Join_Graphs.add_edge(str(trunkpoint5.point), str(trunkpoint6.point), object=Edge(trunkpoint5, trunkpoint6))
# Join_Graphs.add_edge(str(trunkpoint6.point), str(trunkpoint7.point), object=Edge(trunkpoint6, trunkpoint7))
# Join_Graphs.add_edge(str(trunkpoint7.point), str(trunkpoint8.point), object=Edge(trunkpoint7, trunkpoint8))
# Join_Graphs.add_edge(str(trunkpoint8.point), str(trunkpoint9.point), object=Edge(trunkpoint8, trunkpoint9))
# Join_Graphs.add_edge(str(trunkpoint9.point), str(trunkpoint10.point), object=Edge(trunkpoint9, trunkpoint10))
# Join_Graphs.add_edge(str(trunkpoint10.point), str(trunkpoint11.point), object=Edge(trunkpoint10, trunkpoint11))
# Join_Graphs.add_edge(str(trunkpoint6.point), str(trunkpoint12.point), object=Edge(trunkpoint6, trunkpoint12))
# Join_Graphs.add_edge(str(trunkpoint12.point), str(trunkpoint13.point), object=Edge(trunkpoint12, trunkpoint13))
# Join_Graphs.add_edge(str(trunkpoint13.point), str(trunkpoint14.point), object=Edge(trunkpoint13, trunkpoint14))
# Join_Graphs.add_edge(str(trunkpoint14.point), str(trunkpoint15.point), object=Edge(trunkpoint14, trunkpoint15))
# Join_Graphs.add_edge(str(trunkpoint14.point), str(trunkpoint16.point), object=Edge(trunkpoint14, trunkpoint16))
# Join_Graphs.add_edge(str(trunkpoint16.point), str(trunkpoint17.point), object=Edge(trunkpoint16, trunkpoint17))
#
#
# Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint.point), tuple(trunkpoint1.point))
# Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint1.point), tuple(trunkpoint3.point))
# Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint3.point), tuple(trunkpoint4.point))
# Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint4.point), tuple(trunkpoint5.point))
# Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint5.point), tuple(trunkpoint6.point))
# Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint6.point), tuple(trunkpoint7.point))
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint7.point), tuple(trunkpoint8.point))
# Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint8.point)][str(trunkpoint9.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint8.point)][str(trunkpoint9.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint8.point), tuple(trunkpoint9.point))
# Join_Graphs[str(trunkpoint8.point)][str(trunkpoint9.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint9.point)][str(trunkpoint10.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint9.point)][str(trunkpoint10.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint9.point), tuple(trunkpoint10.point))
# Join_Graphs[str(trunkpoint9.point)][str(trunkpoint10.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint10.point)][str(trunkpoint11.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint10.point)][str(trunkpoint11.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint10.point), tuple(trunkpoint11.point))
# Join_Graphs[str(trunkpoint10.point)][str(trunkpoint11.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint12.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint12.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint6.point), tuple(trunkpoint12.point))
# Join_Graphs[str(trunkpoint6.point)][str(trunkpoint12.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint12.point)][str(trunkpoint13.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint12.point)][str(trunkpoint13.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint12.point), tuple(trunkpoint13.point))
# Join_Graphs[str(trunkpoint12.point)][str(trunkpoint13.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint13.point)][str(trunkpoint14.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint13.point)][str(trunkpoint14.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint13.point), tuple(trunkpoint14.point))
# Join_Graphs[str(trunkpoint13.point)][str(trunkpoint14.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint15.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint15.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint14.point), tuple(trunkpoint15.point))
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint15.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint16.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint16.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint14.point), tuple(trunkpoint16.point))
# Join_Graphs[str(trunkpoint14.point)][str(trunkpoint16.point)]['length'] = distance
#
# Join_Graphs[str(trunkpoint16.point)][str(trunkpoint17.point)]['radius'] = 1
# Join_Graphs[str(trunkpoint16.point)][str(trunkpoint17.point)]['diameter'] = 2
# distance = euclidean_distance(tuple(trunkpoint16.point), tuple(trunkpoint17.point))
# Join_Graphs[str(trunkpoint16.point)][str(trunkpoint17.point)]['length'] = distance
#
# writeGraph(Join_Graphs)
#
#
# import networkx as nx
# from sklearn.cluster import KMeans
# import numpy as np
# import math
#
# # Assuming you have imported the necessary classes and functions
# G = Join_Graphs.copy()
# G1 = Join_Graphs.copy()
# def cluster_neighbors(neighbors):
#     # Initialize a list to save clusters
#     SaveCluster = []
#
#     # Convert the neighbor coordinates to a NumPy array
#     w = [list(map(float, element.strip('[]').split(','))) for element in neighbors]
#     w_array = np.array(w)
#
#     # Create a KMeans clustering model with 2 clusters
#     kmeans = KMeans(n_clusters=2, random_state=0)
#
#     # Fit the model to the data
#     kmeans.fit(w_array)
#
#     # Get cluster labels
#     labels = kmeans.labels_
#
#     # Separate the points into two clusters based on the labels
#     cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
#     cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]
#
#     # Append the clusters to SaveCluster
#     SaveCluster.append(cluster_1)
#     SaveCluster.append(cluster_2)
#
#     return SaveCluster
#
# def calculate_average_and_update_graph(SaveCluster, Join_Graphs, node):
#     save_AVG = []  # Initialize a list to save average coordinates
#     average_nodes = []  # Initialize a list to store average nodes
#     neighbor_lists = []  # Initialize a list to store neighbors of average nodes
#
#     for cluster_index, coordinates in enumerate(SaveCluster):
#         if len(coordinates) > 1:
#             coorNode = eval(node)
#             # Calculate the average point
#             average_coordinate = [
#                 (coorNode[0] + sum(point[0] for point in coordinates)) / (len(coordinates) + 1),
#                 (coorNode[1] + sum(point[1] for point in coordinates)) / (len(coordinates) + 1),
#                 (coorNode[2] + sum(point[2] for point in coordinates)) / (len(coordinates) + 1)
#             ]
#
#             # Print the average point
#             print("Average Point:", average_coordinate)
#             save_AVG.append(average_coordinate)
#
#             # Add the average coordinate to the graph
#             average_vertex = Vertex(average_coordinate)
#             currnode = Vertex(eval(node))
#
#             G.add_node(str(average_vertex.point), pos=average_vertex.point)
#             G.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
#             Join_Graphs.add_node(str(average_vertex.point), pos=average_vertex.point)
#             Join_Graphs.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
#
#             Connect_edges = list(Join_Graphs.edges(node))
#             sav_radius_clustering = []
#
#             for coord in coordinates:
#                 key1 = str(coord)
#                 key2 = str(node)
#
#                 if key1 in Join_Graphs[key2] and 'radius' in Join_Graphs[key2][key1]:
#                     sav_radius_clustering.append(Join_Graphs[key2][key1]['radius'])
#                 elif key2 in Join_Graphs[key1] and 'radius' in Join_Graphs[key1][key2]:
#                     sav_radius_clustering.append(Join_Graphs[key1][key2]['radius'])
#                 else:
#                     # Handle the case where 'radius' is not present
#                     default_radius = 1
#                     Join_Graphs[key2][key1] = {'radius': default_radius, 'diameter': 2}
#                     sav_radius_clustering.append(default_radius)
#
#             averageRad = sum(sav_radius_clustering) / len(sav_radius_clustering)
#             Join_Graphs[node][str(average_coordinate)]['radius'] = averageRad
#             Join_Graphs[node][str(average_coordinate)]['diameter'] = 2 * averageRad
#
#             # Calculate the Euclidean distance between the node and the average point
#             distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
#             Join_Graphs[node][str(average_coordinate)]['length'] = distance
#
#             print(f'Cluster {cluster_index + 1} - Average Coordinate: {average_vertex.point}')
#
#             for t in range(len(coordinates)):
#                 node_SaveCluster = Vertex(coordinates[t])
#                 G.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
#                 Join_Graphs.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
#                 Join_Graphs[str(average_coordinate)][str(coordinates[t])]['radius'] = 0.4*averageRad
#                 Join_Graphs[str(average_coordinate)][str(coordinates[t])]['diameter'] = 2 * (0.4*averageRad)
#                 distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
#                 Join_Graphs[node][str(average_coordinate)]['length'] = distance
#
#             # Save the average node and its neighbors for later processing
#             average_nodes.append(average_vertex.point)
#             neighbor_lists.append(coordinates)
#
#             resulting_clusters_sav = coordinates
#             connected_edges = list(G.edges(node))
#             ##### to check if we can find all edges of the specific node############
#             elements_in_connected_edges = []
#
#             for edge in connected_edges:
#                 if any(str(node) in edge for node in resulting_clusters_sav):
#                     elements_in_connected_edges.append(edge)
#
#             # Convert the elements in resulting_clusters_sav to strings
#             resulting_clusters_sav_str = [str(cluster) for cluster in resulting_clusters_sav]
#
#             # Create the combined list without repetitions
#             combined_list = connected_edges + list(set(
#                 (connected_edge[0], resulting_cluster)
#                 for connected_edge in connected_edges
#                 for resulting_cluster in resulting_clusters_sav_str
#             ))
#
#             if elements_in_connected_edges:
#                 for edge in elements_in_connected_edges:
#                     if G.has_edge(*edge):
#                         G.remove_edge(*edge)
#                     elif G.has_edge(edge[1], edge[0]):
#                         G.remove_edge(edge[1], edge[0])
#
#                     if Join_Graphs.has_edge(*edge):
#                         Join_Graphs.remove_edge(*edge)
#                     elif Join_Graphs.has_edge(edge[1], edge[0]):
#                         Join_Graphs.remove_edge(edge[1], edge[0])
#             else:
#                 for edge in combined_list:
#                     for nbn in resulting_clusters_sav_str:
#                         if str(nbn) in edge:
#                             # Check if the edge or its reverse exists in the graph before removing
#                             if G.has_edge(*edge):
#                                 G.remove_edge(*edge)
#                             elif G.has_edge(edge[1], edge[0]):
#                                 G.remove_edge(edge[1], edge[0])
#                             if Join_Graphs.has_edge(*edge):
#                                 Join_Graphs.remove_edge(*edge)
#                             elif Join_Graphs.has_edge(edge[1], edge[0]):
#                                 Join_Graphs.remove_edge(edge[1], edge[0])
#
#
#
#
#     return average_nodes, neighbor_lists
#
#
# def process_new_node(node, visited):
#     if node in visited:
#         return
#
#     visited.add(node)
#
#     degree = G.degree(node)
#     print(f'Degree of {node}: {degree}')
#
#     if degree > 3:
#         neighbors1 = []
#         outgoing_neighbors = list(G.successors(node))
#         incoming_neighbors = list(G.predecessors(node))
#         if len(outgoing_neighbors) != 0 and len(incoming_neighbors) != 0 and len(outgoing_neighbors) <= degree - 2:
#             neighbors1.append(outgoing_neighbors)
#             neighbors1.append(incoming_neighbors)
#             neighbors1 = [item for sublist in neighbors1 for item in sublist]
#             neighbors1 = [neighbor for neighbor in neighbors1 if neighbor not in map(str, SaveClusterAvg)]
#         elif len(outgoing_neighbors) != 0 or len(incoming_neighbors) != 0 and len(outgoing_neighbors) <= degree - 1:
#             neighbors1 = list(G.neighbors(node))
#
#         resulting_clusters = cluster_neighbors(neighbors1)
#
#         if resulting_clusters:
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
#             print(average_nodes)
#
#             if neighbors:
#                 for neighbor in neighbors:
#                     process_new_node(neighbor, visited)
#
#     elif degree > 3:
#         neighbors1 = list(G.neighbors(node))
#         resulting_clusters = cluster_neighbors(neighbors1)
#
#         if resulting_clusters:
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
#             print(average_nodes)
#
#             if neighbors:
#                 for neighbor in neighbors:
#                     process_new_node(neighbor, visited)
#
# # Create a set to keep track of visited nodes
# visited_nodes = set()
#
# while True:
#     nodes_with_degree_greater_than_3 = [node for node in Join_Graphs.nodes if Join_Graphs.degree(node) > 3]
#     if not nodes_with_degree_greater_than_3:
#         break
#     for node in nodes_with_degree_greater_than_3:
#         if node in visited_nodes:
#             continue
#
#         visited_nodes.add(node)
#
#         degree = Join_Graphs.degree(node)
#         print(f'Degree of {node}: {degree}')
#
#         neighbors = list(Join_Graphs.neighbors(node))
#         resulting_clusters = cluster_neighbors(neighbors)
#
#         if resulting_clusters:
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, Join_Graphs, node)
#             print(average_nodes)
#
#             for neighbor in neighbors:
#                 process_new_node(neighbor, visited_nodes)
#
#
#
#
# writeGraph(Join_Graphs)
#

########################## remove the degree more than 3 from Join graph (old) ###############
# import networkx as nx
# from sklearn.cluster import KMeans
# import numpy as np
# import math
#
# # Assuming you have imported the necessary classes and functions
# Roots1 = [item for sublist in Roots for item in sublist]
# G = Join_Graphs.copy()
# G1 = Join_Graphs.copy()
#
# visitNode = []
#
# def process_node(node, visited):
#     # Check if the node has already been visited
#     if node in visited:
#         return
#
#     # Mark the current node as visited
#     visited.add(node)
#
#     visitNode.append(node)
#
#     degree = G.degree(node)
#     print(f'Degree of {node}: {degree}')
#     if degree >= 3 and eval(node) in Roots1:
#         neighbors = []
#         outgoing_neighbors = list(G.successors(node))
#         incoming_neighbors = list(G.predecessors(node))
#         if len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 2:
#             neighbors.append(outgoing_neighbors)
#             neighbors.append(incoming_neighbors)
#         # if len(incoming_neighbors) != 0:
#         #     neighbors.append(incoming_neighbors)
#             neighbors = [item for sublist in neighbors for item in sublist]
#         # neighbors = list(G.neighbors(node))
#         elif len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 1:
#             neighbors = list(G.neighbors(node))
#         # Create clusters based on neighbors
#         clusters = cluster_neighbors(neighbors)
#
#         if clusters:
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(clusters, G, node)
#             # process_new_node(average_nodes, visited)
#             # Process the newly created nodes and their neighbors
#             for neighbor in neighbors:
#                 process_new_node(neighbor, visited)
#
#     elif degree >3 and eval(node) not in Roots1:
#         # neighbors = []
#         # outgoing_neighbors = list(G.successors(node))
#         # incoming_neighbors = list(G.predecessors(node))
#         # neighbors.append(outgoing_neighbors)
#         # neighbors.append(incoming_neighbors)
#         # neighbors = [item for sublist in neighbors for item in sublist]
#         neighbors = list(G.neighbors(node))
#
#         # Create clusters based on neighbors
#         clusters = cluster_neighbors(neighbors)
#
#         if clusters:
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(clusters, G, node)
#             # process_new_node(average_nodes, visited)
#             # Process the newly created nodes and their neighbors
#             for neighbor in neighbors:
#                 process_new_node(neighbor, visited)
# def cluster_neighbors(neighbors):
#     # Initialize a list to save clusters
#     SaveCluster = []
#
#     # Convert the neighbor coordinates to a NumPy array
#     w = [list(map(float, element.strip('[]').split(','))) for element in neighbors]
#     w_array = np.array(w)
#
#     # Create a KMeans clustering model with 2 clusters
#     kmeans = KMeans(n_clusters=2, random_state=0)
#
#     # Fit the model to the data
#     kmeans.fit(w_array)
#
#     # Get cluster labels
#     labels = kmeans.labels_
#
#     # Separate the points into two clusters based on the labels
#     cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
#     cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]
#
#     # Append the clusters to SaveCluster
#     SaveCluster.append(cluster_1)
#     SaveCluster.append(cluster_2)
#
#     return SaveCluster
#
# def calculate_average_and_update_graph(SaveCluster, G, node):
#     save_AVG = []  # Initialize a list to save average coordinates
#     average_nodes = []  # Initialize a list to store average nodes
#     neighbor_lists = []  # Initialize a list to store neighbors of average nodes
#
#     for cluster_index, coordinates in enumerate(SaveCluster):
#         if len(coordinates) > 1:
#             coorNode = eval(node)
#             # Calculate the average point
#             average_coordinate = [
#                 (coorNode[0] + sum(point[0] for point in coordinates)) / (len(coordinates) + 1),
#                 (coorNode[1] + sum(point[1] for point in coordinates)) / (len(coordinates) + 1),
#                 (coorNode[2] + sum(point[2] for point in coordinates)) / (len(coordinates) + 1)
#             ]
#
#             # Print the average point
#             print("Average Point:", average_coordinate)
#             save_AVG.append(average_coordinate)
#
#             # Add the average coordinate to the graph
#             average_vertex = Vertex(average_coordinate)
#             currnode = Vertex(eval(node))
#
#             G.add_node(str(average_vertex.point), pos=average_vertex.point)
#             G.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
#             Join_Graphs.add_node(str(average_vertex.point), pos=average_vertex.point)
#             Join_Graphs.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
#
#             Connect_edges = list(Join_Graphs.edges(node))
#             sav_radius_clustering = []
#
#             for h in range(len(coordinates)):
#                 if 'radius' in Join_Graphs[node][str(coordinates[h])]:
#                     sav_radius_clustering.append(Join_Graphs[node][str(coordinates[h])]['radius'])
#                 else:
#                     Join_Graphs[node][str(coordinates[h])]['radius'] = 1
#                     Join_Graphs[node][str(coordinates[h])]['diameter'] = 2
#                     sav_radius_clustering.append(Join_Graphs[node][str(coordinates[h])]['radius'])
#
#             averageRad = sum(sav_radius_clustering) / len(sav_radius_clustering)
#             Join_Graphs[node][str(average_coordinate)]['radius'] = averageRad
#             Join_Graphs[node][str(average_coordinate)]['diameter'] = 2 * averageRad
#
#             # Calculate the Euclidean distance between the node and the average point
#             distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
#             Join_Graphs[node][str(average_coordinate)]['length'] = distance
#
#             print(f'Cluster {cluster_index + 1} - Average Coordinate: {average_vertex.point}')
#
#             for t in range(len(coordinates)):
#                 node_SaveCluster = Vertex(coordinates[t])
#                 G.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
#                 Join_Graphs.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
#                 Join_Graphs[str(average_coordinate)][str(coordinates[t])]['radius'] = 0.4*averageRad
#                 Join_Graphs[str(average_coordinate)][str(coordinates[t])]['diameter'] = 2 * (0.4*averageRad)
#                 distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
#                 Join_Graphs[node][str(average_coordinate)]['length'] = distance
#
#             # Save the average node and its neighbors for later processing
#             average_nodes.append(average_vertex.point)
#             neighbor_lists.append(coordinates)
#
#             resulting_clusters_sav = coordinates
#             connected_edges = list(G.edges(node))
#
#
#             for edge in connected_edges:
#                 for nbn in resulting_clusters_sav:
#                     if str(nbn) in edge:
#                         G.remove_edge(*edge)
#                         Join_Graphs.remove_edge(*edge)
#     return average_nodes, neighbor_lists
#
# # Define the Euclidean distance function
# def euclidean_distance(point1, point2):
#     if len(point1) != 3 or len(point2) != 3:
#         raise ValueError("Both points should be 3D coordinates.")
#
#     x1, y1, z1 = point1
#     x2, y2, z2 = point2
#
#     distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
#
#     return distance
#
# # Define the process_new_node function
# def process_new_node(node, visited, neighbors=None):
#     if node in visited:
#         return
#
#     visited.add(node)
#
#     degree = G.degree(node)
#     print(f'Degree of {node}: {degree}')
#
#     if degree >= 3 and eval(node) in Roots1:
#         # Handle nodes with degree > 3
#         print(f'Processing new node with degree > 3: {node}')
#
#         # Add your logic here for processing new nodes with degree > 3
#         # You can create the average point, add it to the graph, and add edges
#
#
#         neighbors1 = []
#         outgoing_neighbors = list(G.successors(node))
#         incoming_neighbors = list(G.predecessors(node))
#         if len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 2:
#             neighbors1.append(outgoing_neighbors)
#             neighbors1.append(incoming_neighbors)
#         # if len(incoming_neighbors) != 0:
#         #     neighbors1.append(incoming_neighbors)
#             neighbors1 = [item for sublist in neighbors1 for item in sublist]
#         # neighbors1 = list(G.neighbors(node))
#
#         elif len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 1:
#             neighbors1 = list(G.neighbors(node))
#         resulting_clusters = cluster_neighbors(neighbors1)
#
#         if resulting_clusters:
#             # For example, you can use your existing code to create the average point and add it to the graph
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
#             print(average_nodes)
#
#             # After adding the new node and edges, you can process the neighbors
#             if neighbors:
#                 for neighbor in neighbors:
#                     process_new_node(neighbor, visited)
#
#
#     elif degree >3 and eval(node) not in Roots1:
#          # Handle nodes with degree > 3
#         print(f'Processing new node with degree > 3: {node}')
#
#         # Add your logic here for processing new nodes with degree > 3
#         # You can create the average point, add it to the graph, and add edges
#
#
#         # neighbors1 = []
#         # outgoing_neighbors = list(G.successors(node))
#         # incoming_neighbors = list(G.predecessors(node))
#         # neighbors1.append(outgoing_neighbors)
#         # neighbors1.append(incoming_neighbors)
#         # neighbors1 = [item for sublist in neighbors1 for item in sublist]
#         neighbors1 = list(G.neighbors(node))
#         resulting_clusters = cluster_neighbors(neighbors1)
#
#         if resulting_clusters:
#             # For example, you can use your existing code to create the average point and add it to the graph
#             average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
#             print(average_nodes)
#
#             # After adding the new node and edges, you can process the neighbors
#             if neighbors:
#                 for neighbor in neighbors:
#                     process_new_node(neighbor, visited)
#
# # Create a set to keep track of visited nodes
# visited_nodes = set()
#
# # # Process the root node
# # process_node(root_node, visited_nodes)
#
# # # Process other nodes in the graph
# # for ind, gnode in enumerate(G1.nodes):
# #     if gnode not in visitNode:
# #         process_node(gnode, visited_nodes)
# #     # else:
# #     #     if gnode not in visitNode:
# #     #         # Start the traversal from the root node
# #     #         process_node(gnode, visited_nodes)
# # # Add a loop to continue processing nodes with degree > 3
# while True:
#     nodes_with_degree_greater_than_3 = [node for node in G.nodes if G.degree(node) >= 3]
#     if not nodes_with_degree_greater_than_3:
#         break
#     for node in nodes_with_degree_greater_than_3:
#         process_node(node, visited_nodes)
# ###############################################################################################
# for b in range(len(Roots)):
#     Root_points = Roots[b]
#     for rootN in Root_points:
# # rootN = '[145.470187956702, 139.918452882767, 196.563]'
#         bfs_tree = nx.bfs_tree(Join_Graphs, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)
#
#         # Find the parent of each node
#         parent = {}
#         for node in bfs_tree.nodes():
#             parent_nodes = list(bfs_tree.predecessors(node))
#             if parent_nodes:
#                 parent[node] = parent_nodes[0]
#             else:
#                 parent[node] = None
#
#
#         for edge in bfs_tree.edges():
#             parent[edge[1]] = edge[0]
#
#
#         # Print the parents of each node
#         print("Parents of each node:", parent)
#
#         check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
#         node_list = list(bfs_tree.nodes())
#         edge_list = list(bfs_tree.edges())
#
#         check = []
#         ratiocheck = {}
#         Diamcheck = False
#         for node in bfs_tree.nodes():
#
#             node = str(node)
#             # node_index = node_list.index(node)
#             node_index = list(bfs_tree.nodes()).index(node)
#             if check_node[node_index] == False:
#
#                 if parent[node] != None:
#
#                         par = Join_Graphs[parent[node]][node]["radius"]
#
#                         if len(list(Join_Graphs.neighbors(node))) > 0:
#                             nnb = list(Join_Graphs.neighbors(node))
#
#                             # if str(rootN) in nnb:
#                             #     nnb.remove(str(rootN))
#                             queue = []
#                             for j in range(len(nnb)):
#                                 nnb_index = list(bfs_tree.nodes()).index(nnb[j])
#                                 if check_node[nnb_index] == False:
#                                     if par!= 0:
#
#                                         # children = nnb
#                                         child = Join_Graphs[node][nnb[j]]["radius"]
#                                         ratioDiam = (child / par)
#
#                                         check_node[node_index] = True
#                                         check.append(node)
#                                         # node_index = node_list.index(nnb[j])
#
#                                         # check_node[nnb_index] = True
#                                         if ratioDiam >1:
#
#                                             Join_Graphs[node][nnb[j]]["radius"] = 0.4 * par
#                                             Join_Graphs[node][nnb[j]]["diameter"] = 2*(0.4 * par)
#
#                                             par = Join_Graphs[parent[node]][node]["diameter"]
#                                             child = Join_Graphs[node][nnb[j]]["diameter"]
#                                             ratioDiam = (child / par)
#                                             Join_Graphs[node][nnb[j]]["ratioDiam"] = ratioDiam

#############################################################################################


writeGraph(Join_Graphs)


save_closest_node = []
save_Avg_trunkpoint_Rootpoint_eachlobe = []
save_Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe = []
for v in range(len(Roots)):
    Root_points = Roots[v]
    # trunkpoint = [137, 190.264, -88.356] #[127.131, 198.92, -94.571]
    trunkpoint = [145.470187956702, 139.918452882767, 196.563] #sub1
    # trunkpoint = [158.011, 153.247, 201.52] # sub2
    # trunkpoint = [158.011, 150.247, 191.52] #sub3, sub4
    # trunkpoint = [153.011, 145.247, 208.52] #sub5
    # trunkpoint = [153.011, 161.247, 163.52]#sub6
    # trunkpoint = [14.825, -153.636, -94.129]#sub7
    # trunkpoint = [-1.989, -149.753, -149.48]#sub8
    # trunkpoint = [-5.175, -129.636, -200.129] #sub9
    # trunkpoint = [-15.175, 2.364, -156.129] #sub10
    ###################### find the root points for each lobe ######################

    #################### create two cluster for Rootpoint_rul##############################
    from sklearn.cluster import KMeans
    import numpy as np
    save_cluster = []

    # List of 3D points
    b = Root_points
    if len(b) > 1:
        # Convert the list to a NumPy array
        b_array = np.array(b)
        # Create a KMeans clustering model with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0)
        # Fit the model to the data
        kmeans.fit(b_array)
        # Get cluster labels
        labels = kmeans.labels_
        # Separate the points into two clusters based on the labels
        cluster_1 = [b[i] for i in range(len(b)) if labels[i] == 0]
        cluster_2 = [b[i] for i in range(len(b)) if labels[i] == 1]
        save_cluster.append(cluster_1)
        save_cluster.append(cluster_2)
        # Print the two clusters
        print("Cluster 1:")
        for point in cluster_1:
            print(point)
        print("\nCluster 2:")
        for point in cluster_2:
            print(point)
    else:
        save_cluster.append(b)
    ######################################### find the closest point in each cluster to the trunkpoint ####################


    closer_point = []
    for h in range(len(save_cluster)):
        coordinates = save_cluster[h]
        # Initialize variables to store the closest node and its distance
        closest_node = None
        min_distance = float('inf')  # Initialize to positive infinity
        # Calculate the distance between the trunkpoint and each node
        for coord in coordinates:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(trunkpoint, coord)))

            # Check if this node is closer
            if distance < min_distance:
                min_distance = distance
                closest_node = coord

        closer_point.append(closest_node)
        print("Closest Node:", closest_node)
        print("Minimum Distance:", min_distance)

    ########################### calculate average closer_points with trunkpoint
    Avg_trunkpoint_Rootpoint_eachlobe = [
        (closer_point[0][0] + trunkpoint[0]) / 2,
        (closer_point[0][1] + trunkpoint[1]) / 2,
        (closer_point[0][2] + trunkpoint[2]) / 2
    ]
    print("Midpoint between list 'a' and 'node' (x, y, z):")
    print(Avg_trunkpoint_Rootpoint_eachlobe)
    save_Avg_trunkpoint_Rootpoint_eachlobe.append(Avg_trunkpoint_Rootpoint_eachlobe)

    ########################## calculate average Avg_trunkpoint_Rootpoint_eachlobe with closer points
    Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe = [
        (closer_point[0][0] + Avg_trunkpoint_Rootpoint_eachlobe[0]) / 2,
        (closer_point[0][1] + Avg_trunkpoint_Rootpoint_eachlobe[1]) / 2,
        (closer_point[0][2] + Avg_trunkpoint_Rootpoint_eachlobe[2]) / 2
    ]
    print("Midpoint between list 'a' and 'node' (x, y, z):")
    print(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe)
    save_Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.append(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe)


    # ##########################label graph_in_group2#############
    # g3 = Final_graph_sidebranch_group2.copy()
    # components = list(nx.weakly_connected_components(g3))
    #
    # # Create a dictionary to store component labels
    # component_labels = {}
    #
    # # Assign labels to each component
    # for i, component in enumerate(components):
    #     for node in component:
    #         component_labels[node] = i + 1
    #
    # # Print the component labels
    # for node, label in component_labels.items():
    #     print(f"Node {node}: Component {label}")
    # unique_labels = set(component_labels.values())
    # label_array = np.array(list(unique_labels))
    # # Print the unique labels
    # print("Unique Labels:")
    # for label in unique_labels:
    #     print(label)
    #
    # selected_label = len(unique_labels)
    #
    # label_root_store = {}
    #
    #
    # for i in range(selected_label):
    #     nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
    #     edges_in_label = [edge for edge in g3.edges() if
    #                       component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]
    #
    #     subgraph1 = nx.DiGraph()
    #
    #     # Convert your 'a' list to a list of lists
    #     a = [[float(coord) for coord in point.strip('[]').split(', ')] for point in nodes_in_label]
    #
    #     # Convert 'Rootpoint_lul' to a similar format
    #     Root_points = [[float(coord) for coord in point] for point in Root_points]
    #
    #     # Initialize a list to store matching elements
    #     matching_elements = []
    #
    #     # Iterate through 'Rootpoint_lul' and 'a' to find matches
    #     for point in Root_points:
    #         if point in a:
    #             matching_elements.append(point)
    #
    #     # Print the matching elements
    #     print("Matching elements in 'Rootpoint_lul':")
    #     for element in matching_elements:
    #         print(element)
    #
    #
    #     # coor_set = set(tuple(point) for point in Rootpoint_rul)
    #     if len(matching_elements) > 0:
    #         for node in nodes_in_label:
    #             # if tuple(eval(node)) in coor_set:
    #                 subgraph1.add_node(node)
    #         # Iterate through the edges in Join_Graphs and add edges that connect nodes in the subgraph
    #         for edge in edges_in_label:
    #             node1, node2 = edge
    #             if node1 in subgraph1.nodes and node2 in subgraph1.nodes:
    #                 subgraph1.add_edge(node1, node2)
    #         #########################label subgraph1 ##################
    #         visitNode = []
    #         G = subgraph1.copy()
    #         G1 = subgraph1.copy()
    #         def process_node(node, visited):
    #             # Check if the node has already been visited
    #             if node in visited:
    #                 return
    #
    #             # Mark the current node as visited
    #             visited.add(node)
    #
    #             visitNode.append(node)
    #
    #             degree = G.degree(node)
    #             print(f'Degree of {node}: {degree}')
    #             if degree >3:
    #
    #                 def cluster_neighbors(neighbors):
    #                     # Initialize a list to save clusters
    #                     SaveCluster = []
    #
    #                     # Convert the neighbor coordinates to a NumPy array
    #                     w = [list(map(float, element.strip('[]').split(','))) for element in neighbors]
    #                     w_array = np.array(w)
    #
    #                     # Create a KMeans clustering model with 2 clusters
    #                     kmeans = KMeans(n_clusters=2, random_state=0)
    #
    #                     # Fit the model to the data
    #                     kmeans.fit(w_array)
    #
    #                     # Get cluster labels
    #                     labels = kmeans.labels_
    #
    #                     # Separate the points into two clusters based on the labels
    #                     cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
    #                     cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]
    #
    #                     # Append the clusters to SaveCluster
    #                     SaveCluster.append(cluster_1)
    #                     SaveCluster.append(cluster_2)
    #
    #                     return SaveCluster
    #
    #                 def calculate_average_and_update_graph(SaveCluster, G, node):
    #                     save_AVG = []  # Initialize a list to save average coordinates
    #                     average_nodes = []  # Initialize a list to store average nodes
    #                     neighbor_lists = []  # Initialize a list to store neighbors of average nodes
    #
    #
    #                     for cluster_index, coordinates in enumerate(SaveCluster):
    #                         if len(coordinates)>1:
    #
    #
    #                             coorNode = eval(node)
    #                             # Calculate the average point
    #                             average_coordinate = [
    #                                 (coorNode[0] + sum(point[0] for point in coordinates)) / (len(coordinates) + 1),
    #                                 (coorNode[1] + sum(point[1] for point in coordinates)) / (len(coordinates) + 1),
    #                                 (coorNode[2] + sum(point[2] for point in coordinates)) / (len(coordinates) + 1)
    #                             ]
    #
    #                             # Print the average point
    #                             print("Average Point:", average_coordinate)
    #                             save_AVG.append(average_coordinate)
    #
    #                             # Add the average coordinate to the graph
    #                             average_vertex = Vertex(average_coordinate)
    #                             currnode = Vertex(eval(node))
    #                             Join_Graphs.add_node(str(average_vertex.point), pos=average_vertex.point)
    #                             Join_Graphs.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
    #                             G.add_node(str(average_vertex.point), pos=average_vertex.point)
    #                             G.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
    #                             print(f'Cluster {cluster_index + 1} - Average Coordinate: {average_vertex.point}')
    #                             for t in range(len(coordinates)):
    #                                 node_SaveCluster = Vertex(coordinates[t])
    #                                 Join_Graphs.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
    #
    #
    #                             # Save the average node and its neighbors for later processing
    #                             average_nodes.append(average_vertex.point)
    #                             neighbor_lists.append(coordinates)
    #
    #                             resulting_clusters_sav = coordinates
    #                             connected_edges = list(G.edges(node))
    #                             for edge in range(len(connected_edges)):
    #                                 for nbn in range(len(resulting_clusters_sav)):
    #                                     if str(resulting_clusters_sav[nbn]) in connected_edges[edge]:
    #                                         G.remove_edge(*connected_edges[edge])
    #                                         Join_Graphs.remove_edge(*connected_edges[edge])
    #
    #                     return average_nodes, neighbor_lists
    #
    #
    #
    #                 neighbors1 = list(G.neighbors(node))
    #                 resulting_clusters = cluster_neighbors(neighbors1)
    #
    #                 for s in range(len(resulting_clusters)):
    #                     if len(resulting_clusters[s]) >1:
    #                         average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
    #                         # print(average_nodes)
    #                         def process_new_node(node, visited, neighbors=None):
    #                             if node in visited:
    #                                 return
    #
    #                             visited.add(node)
    #
    #                             degree = G.degree(node)
    #                             print(f'Degree of {node}: {degree}')
    #
    #                             if degree > 3:
    #                                 # Handle nodes with degree > 3
    #                                 print(f'Processing new node with degree > 3: {node}')
    #
    #                                 # Add your logic here for processing new nodes with degree > 3
    #                                 # You can create the average point, add it to the graph, and add edges
    #
    #                                 neighbors1 = list(G.neighbors(node))
    #                                 resulting_clusters = cluster_neighbors(neighbors1)
    #                                 # For example, you can use your existing code to create the average point and add it to the graph
    #                                 average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
    #                                 print(average_nodes)
    #
    #                                 # After adding the new node and edges, you can process the neighbors
    #                                 if neighbors:
    #                                     for neighbor in neighbors:
    #                                         process_new_node(str(neighbor), visited)
    #
    #
    #
    #
    #             # Get the neighbors of the current node
    #             neighbors = list(G.neighbors(node))
    #
    #             for neighbor in neighbors:
    #                 process_node(neighbor, visited)
    #
    #
    #         for subgraph_node in subgraph1.nodes:
    #             if eval(subgraph_node) in Root_points:
    #                 # Define your root node
    #                 root_node = subgraph_node
    #
    #         # Create a set to keep track of visited nodes
    #         visited_nodes = set()
    #
    #
    #         for ind, gnode  in enumerate(G1.nodes):
    #             if ind == 0:
    #                 process_node(root_node, visited_nodes)
    #             else:
    #                 if gnode not in visitNode:
    #                     # Start the traversal from the root node
    #                     process_node(gnode, visited_nodes)




    ###################### label Join_graph ##################
    g = Join_Graphs.copy()
    # g = Final_graph_sidebranch_group2.copy()
    components = list(nx.weakly_connected_components(g))

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    label_root_store = {}
    subgraph = nx.DiGraph()

    for i in range(selected_label):
        nodes_in_label = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label = [edge for edge in g.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]


        # Convert your 'a' list to a list of lists
        a = [[float(coord) for coord in point.strip('[]').split(', ')] for point in nodes_in_label]

        # Convert 'Rootpoint_lul' to a similar format
        Root_points = [[float(coord) for coord in point] for point in Root_points]

        # Initialize a list to store matching elements
        matching_elements = []

        # Iterate through 'Rootpoint_lul' and 'a' to find matches
        for point in Root_points:
            if point in a:
                matching_elements.append(point)

        # Print the matching elements
        print("Matching elements in 'Rootpoint_lul':")
        for element in matching_elements:
            print(element)


        # coor_set = set(tuple(point) for point in Rootpoint_rul)
        if len(matching_elements) > 0:
            for node in nodes_in_label:
                # if tuple(eval(node)) in coor_set:
                    subgraph.add_node(node)
            # Iterate through the edges in Join_Graphs and add edges that connect nodes in the subgraph
            for edge in edges_in_label:
                node1, node2 = edge
                if node1 in subgraph.nodes and node2 in subgraph.nodes:
                    subgraph.add_edge(node1, node2)


    ######################### label subgraph ####################

    g1 = subgraph.copy()
    components = list(nx.weakly_connected_components(g1))

    # Create a dictionary to store component labels
    component_labels = {}

    # Assign labels to each component
    for i, component in enumerate(components):
        for node in component:
            component_labels[node] = i + 1

    # Print the component labels
    for node, label in component_labels.items():
        print(f"Node {node}: Component {label}")
    unique_labels = set(component_labels.values())
    label_array = np.array(list(unique_labels))
    # Print the unique labels
    print("Unique Labels:")
    for label in unique_labels:
        print(label)

    selected_label = len(unique_labels)

    closest_nodes=[]

    for i in range(selected_label):
        nodes_in_label1 = [node for node, label in component_labels.items() if label == label_array[i]]
        edges_in_label1 = [edge for edge in g1.edges() if
                          component_labels[edge[0]] == label_array[i] and component_labels[edge[1]] == label_array[i]]



        # Convert 'nodes_in_label' to a list of lists
        nodes_in_label1 = [[float(coord) for coord in point.strip('[]').split(', ')] for point in nodes_in_label1]

        # Convert 'node1' to a numpy array for easy calculation
        node1 = np.array(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe)

        # Initialize variables to keep track of the closest element and its distance
        closest_element = None
        closest_distance = float('inf')

        # Iterate through 'nodes_in_label' to find the closest element
        for element in nodes_in_label1:
            if element in Root_points:
                element_coords = np.array(element)
                distance = np.linalg.norm(node1 - element_coords)  # Euclidean distance
                if distance < closest_distance:
                    closest_node = element
                    closest_distance = distance


        # Print the closest element and its distance
        print("Closest element to 'node1':", closest_node)
        print("Distance to 'node1':", closest_distance)
        closest_nodes.append(closest_node)
    save_closest_node.append(closest_nodes)


    ##################################################### add new nodes and elements to Join_Graph
    if str(trunkpoint) not in Join_Graphs:
        trunkpoint = Vertex(trunkpoint)
        Join_Graphs.add_node(str(trunkpoint.point), pos=trunkpoint.point)
    # Avg_trunkpoint_Rootpoint_eachlobe = Vertex(Avg_trunkpoint_Rootpoint_eachlobe)
    Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe = Vertex(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe)


    for z in range(len(closest_nodes)):
        clos_node = closest_nodes[z]
        clos_node = Vertex(clos_node)
        Join_Graphs.add_edge(str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point), str(clos_node.point), object=Edge(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe, clos_node))
        Join_Graphs[str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point)][str(clos_node.point)]['radius'] = 1
        Join_Graphs[str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point)][str(clos_node.point)]['diameter'] = 2
        # distance = euclidean_distance(tuple(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point), tuple(clos_node.point))
        # Final_graph_sidebranch_group2[str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point)][str(clos_node.point)]['length'] = distance
    # Join_Graphs.add_node(str(trunkpoint1.point), pos=trunkpoint1.point)
    # Join_Graphs.add_node(str(Avg_trunkpoint_Rootpoint_eachlobe.point), pos=Avg_trunkpoint_Rootpoint_eachlobe.point)
    Join_Graphs.add_node(str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point), pos=Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point)

    # subgraph.add_edge(str(trunkpoint.point), str(Avg_trunkpoint_Rootpoint_eachlobe.point), object=Edge(trunkpoint, Avg_trunkpoint_Rootpoint_eachlobe))
    # subgraph.add_edge(str(Avg_trunkpoint_Rootpoint_eachlobe.point), str(Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe.point), object=Edge(Avg_trunkpoint_Rootpoint_eachlobe, Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe))







############################# clustring save_Avg_trunkpoint_Rootpoint_eachlobe in two categories
SaveCluster = []
# List of 3D points
z = save_Avg_trunkpoint_Rootpoint_eachlobe
# Convert the list to a NumPy array
z_array = np.array(z)
# Create a KMeans clustering model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
# Fit the model to the data
kmeans.fit(z_array)
# Get cluster labels
labels = kmeans.labels_
# Separate the points into two clusters based on the labels
cluster_1 = [z[i] for i in range(len(z)) if labels[i] == 0]
cluster_2 = [z[i] for i in range(len(z)) if labels[i] == 1]
SaveCluster.append(cluster_1)
SaveCluster.append(cluster_2)
# Print the two clusters
print("Cluster 1:")
for point in cluster_1:
    print(point)
print("\nCluster 2:")
for point in cluster_2:
    print(point)

#################################### find the average point in SaveCluster,and connect avgpoint to trunk point in Join graph
save_AVG=[]
for n in range(len(SaveCluster)):
    coordinates = SaveCluster[n]

    # Initialize variables to store the sum of coordinates
    sum_x, sum_y, sum_z = 0, 0, 0

    # Iterate through the coordinates and calculate the sum
    for point in coordinates:
        sum_x += point[0]
        sum_y += point[1]
        sum_z += point[2]

    # Calculate the average
    num_points = len(coordinates)
    average_x = sum_x / num_points
    average_y = sum_y / num_points
    average_z = sum_z / num_points

    # Create the average coordinate as a list
    average_coordinate = [average_x, average_y, average_z]
    save_AVG.append(average_coordinate)
    average_coordinate = Vertex(average_coordinate)
    trunkpoint = Vertex(trunkpoint)
    Join_Graphs.add_node(str(average_coordinate.point), pos=average_coordinate.point)
    Join_Graphs.add_edge(str(trunkpoint.point), str(average_coordinate.point), object=Edge(trunkpoint, average_coordinate))
    Join_Graphs[str(trunkpoint.point)][str(average_coordinate.point)]['radius'] = 1
    Join_Graphs[str(trunkpoint.point)][str(average_coordinate.point)]['diameter'] = 2
    # distance = euclidean_distance(tuple(trunkpoint.point), tuple(average_coordinate.point))
    # Final_graph_sidebranch_group2[str(trunkpoint.point)][str(average_coordinate.point)]['length'] = distance
    print(average_coordinate)



############################### clustring save_Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe in two categories

SaveClusterAvg = []
# List of 3D points
w = save_Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe
# Convert the list to a NumPy array
w_array = np.array(w)
# Create a KMeans clustering model with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
# Fit the model to the data
kmeans.fit(w_array)
# Get cluster labels
labels = kmeans.labels_
# Separate the points into two clusters based on the labels
cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]
SaveClusterAvg.append(cluster_1)
SaveClusterAvg.append(cluster_2)
# Print the two clusters
print("Cluster 1:")
for point in cluster_1:
    print(point)
print("\nCluster 2:")
for point in cluster_2:
    print(point)

########################### find which cluster in save_Avg_Rootpoints_with_Avg_trunkpoint_Rootpoint_eachlobe closer to average_coordinate
import math
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
def find_closer_list(node, list1, list2):
    centroid_list1 = [sum(x) / len(list1) for x in zip(*list1)]
    centroid_list2 = [sum(x) / len(list2) for x in zip(*list2)]

    distance_to_list1 = euclidean_distance(node, centroid_list1)
    distance_to_list2 = euclidean_distance(node, centroid_list2)

    if distance_to_list1 < distance_to_list2:
        return "list1"
    else:
        return "list2"


list1 = SaveClusterAvg[0]
list2 = SaveClusterAvg[1]
for i, node in enumerate(save_AVG):
    closer_list = find_closer_list(node, list1, list2)
    print(f"Node {i+1} is closer to {closer_list}. Node: {node}")
    node = Vertex(node)
    if closer_list == 'list1':
       nodes_closer_list =  list1
    elif closer_list == 'list2':
        nodes_closer_list = list2
    for c in range(len(nodes_closer_list)):
        node_closer_list = Vertex(nodes_closer_list[c])
        Join_Graphs.add_edge(str(node.point), str(node_closer_list.point), object=Edge(node, node_closer_list))
        Join_Graphs[str(node.point)][str(node_closer_list.point)]['radius'] = 1
        Join_Graphs[str(node.point)][str(node_closer_list.point)]['diameter'] = 2
        distance = euclidean_distance(tuple(node.point), tuple(node_closer_list.point))
        Join_Graphs[str(node.point)][str(node_closer_list.point)]['length'] = distance
#sub1
trunkpoint8 = [145.470187956702, 139.918452882767, 126.563]
trunkpoint7 = [145.470187956702, 139.918452882767, 136.563]
trunkpoint6 = [145.470187956702, 139.918452882767, 146.563]
trunkpoint5 = [145.470187956702, 139.918452882767, 156.563]
trunkpoint4 = [145.470187956702, 139.918452882767, 166.563]
trunkpoint3 = [145.470187956702, 139.918452882767, 176.563]
trunkpoint1 = [145.470187956702, 139.918452882767, 186.563]
trunkpoint = [145.470187956702, 139.918452882767, 196.563]
#sub2
# trunkpoint = [158.011, 153.247, 131.52]
# trunkpoint1 = [158.011, 153.247, 141.52]
# trunkpoint3 = [158.011, 153.247, 151.52]
# trunkpoint4 = [158.011, 153.247, 161.52]
# trunkpoint5 = [158.011, 153.247, 171.52]
# trunkpoint6 = [158.011, 153.247, 181.52]
# trunkpoint7 = [158.011, 153.247, 191.52]
# trunkpoint8= [158.011, 153.247, 201.52]
#sub3, sub4
# trunkpoint = [158.011, 150.247, 121.52]
# trunkpoint1 = [158.011, 150.247, 131.52]
# trunkpoint3 = [158.011, 150.247, 141.52]
# trunkpoint4 = [158.011, 150.247, 151.52]
# trunkpoint5 = [158.011, 150.247, 161.52]
# trunkpoint6 = [158.011, 150.247, 171.52]
# trunkpoint7 = [158.011, 150.247, 181.52]
# trunkpoint8 = [158.011, 150.247, 191.52]
#sub5
# trunkpoint = [153.011, 145.247, 138.52]
# trunkpoint1 = [153.011, 145.247, 148.52]
# trunkpoint3 = [153.011, 145.247, 158.52]
# trunkpoint4 = [153.011, 145.247, 168.52]
# trunkpoint5 = [153.011, 145.247, 178.52]
# trunkpoint6 = [153.011, 145.247, 188.52]
# trunkpoint7 = [153.011, 145.247, 198.52]
# trunkpoint8 = [153.011, 145.247, 208.52]
#sub6
# trunkpoint = [153.011, 161.247, 93.52]
# trunkpoint1 = [153.011, 161.247, 103.52]
# trunkpoint3 = [153.011, 161.247, 113.52]
# trunkpoint4 = [153.011, 161.247, 123.52]
# trunkpoint5 = [153.011, 161.247, 133.52]
# trunkpoint6 = [153.011, 161.247, 143.52]
# trunkpoint7 = [153.011, 161.247, 153.52]
# trunkpoint8 = [153.011, 161.247, 163.52]
#sub7
# trunkpoint = [14.825, -153.636, -164.129]
# trunkpoint1 = [14.825, -153.636, -154.129]
# trunkpoint3 = [14.825, -153.636, -144.129]
# trunkpoint4 = [14.825, -153.636, -134.129]
# trunkpoint5 = [14.825, -153.636, -124.129]
# trunkpoint6 = [14.825, -153.636, -114.129]
# trunkpoint7 = [14.825, -153.636, -104.129]
# trunkpoint8 = [14.825, -153.636, -94.129]
# #sub8
# trunkpoint = [-1.989, -149.753, -219.48]
# trunkpoint1 = [-1.989, -149.753, -209.48]
# trunkpoint3 = [-1.989, -149.753, -199.48]
# trunkpoint4 = [-1.989, -149.753, -189.48]
# trunkpoint5 = [-1.989, -149.753, -179.48]
# trunkpoint6 = [-1.989, -149.753, -169.48]
# trunkpoint7 = [-1.989, -149.753, -159.48]
# trunkpoint8 = [-1.989, -149.753, -149.48]
#sub9
# trunkpoint = [-5.175, -129.636, -270.129]
# trunkpoint1 = [-5.175, -129.636, -260.129]
# trunkpoint3 = [-5.175, -129.636, -250.129]
# trunkpoint4 = [-5.175, -129.636, -240.129]
# trunkpoint5 = [-5.175, -129.636, -230.129]
# trunkpoint6 = [-5.175, -129.636, -220.129]
# trunkpoint7 = [-5.175, -129.636, -210.129]
# trunkpoint8 = [-5.175, -129.636, -200.129]

#sub10
# trunkpoint = [-15.175, 2.364, -226.129]
# trunkpoint1 = [-15.175, 2.364, -216.129]
# trunkpoint3 = [-15.175, 2.364, -206.129]
# trunkpoint4 = [-15.175, 2.364, -196.129]
# trunkpoint5 = [-15.175, 2.364, -186.129]
# trunkpoint6 = [-15.175, 2.364, -176.129]
# trunkpoint7 = [-15.175, 2.364, -166.129]
# trunkpoint8 = [-15.175, 2.364, -156.129]

trunkpoint = Vertex(trunkpoint)
trunkpoint1 = Vertex(trunkpoint1)
trunkpoint3 = Vertex(trunkpoint3)
trunkpoint4 = Vertex(trunkpoint4)
trunkpoint5 = Vertex(trunkpoint5)
trunkpoint6 = Vertex(trunkpoint6)
trunkpoint7 = Vertex(trunkpoint7)
trunkpoint8 = Vertex(trunkpoint8)

Join_Graphs.add_node(str(trunkpoint1.point), pos=trunkpoint1.point)
Join_Graphs.add_node(str(trunkpoint3.point), pos=trunkpoint3.point)
Join_Graphs.add_node(str(trunkpoint4.point), pos=trunkpoint4.point)
Join_Graphs.add_node(str(trunkpoint5.point), pos=trunkpoint5.point)
Join_Graphs.add_node(str(trunkpoint6.point), pos=trunkpoint6.point)
Join_Graphs.add_node(str(trunkpoint7.point), pos=trunkpoint7.point)
Join_Graphs.add_node(str(trunkpoint8.point), pos=trunkpoint8.point)

Join_Graphs.add_edge(str(trunkpoint.point), str(trunkpoint1.point), object=Edge(trunkpoint, trunkpoint1))
Join_Graphs.add_edge(str(trunkpoint1.point), str(trunkpoint3.point), object=Edge(trunkpoint1, trunkpoint3))
Join_Graphs.add_edge(str(trunkpoint3.point), str(trunkpoint4.point), object=Edge(trunkpoint3, trunkpoint4))
Join_Graphs.add_edge(str(trunkpoint4.point), str(trunkpoint5.point), object=Edge(trunkpoint4, trunkpoint5))
Join_Graphs.add_edge(str(trunkpoint5.point), str(trunkpoint6.point), object=Edge(trunkpoint5, trunkpoint6))
Join_Graphs.add_edge(str(trunkpoint6.point), str(trunkpoint7.point), object=Edge(trunkpoint6, trunkpoint7))
Join_Graphs.add_edge(str(trunkpoint7.point), str(trunkpoint8.point), object=Edge(trunkpoint7, trunkpoint8))

Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['radius'] = 1
Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint.point), tuple(trunkpoint1.point))
Join_Graphs[str(trunkpoint.point)][str(trunkpoint1.point)]['length'] = distance

Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['radius'] = 1
Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint1.point), tuple(trunkpoint3.point))
Join_Graphs[str(trunkpoint1.point)][str(trunkpoint3.point)]['length'] = distance

Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['radius'] = 1
Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint3.point), tuple(trunkpoint4.point))
Join_Graphs[str(trunkpoint3.point)][str(trunkpoint4.point)]['length'] = distance

Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['radius'] = 1
Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint4.point), tuple(trunkpoint5.point))
Join_Graphs[str(trunkpoint4.point)][str(trunkpoint5.point)]['length'] = distance

Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['radius'] = 1
Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint5.point), tuple(trunkpoint6.point))
Join_Graphs[str(trunkpoint5.point)][str(trunkpoint6.point)]['length'] = distance

Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['radius'] = 1
Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint.point), tuple(trunkpoint1.point))
Join_Graphs[str(trunkpoint6.point)][str(trunkpoint7.point)]['length'] = distance

Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['radius'] = 1
Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['diameter'] = 2
distance = euclidean_distance(tuple(trunkpoint.point), tuple(trunkpoint1.point))
Join_Graphs[str(trunkpoint7.point)][str(trunkpoint8.point)]['length'] = distance
writeGraph(Join_Graphs)


############################################## end new idea ########################
###################################### remove nodes with degree>3 ##################
save_closest_node = [item for sublist in save_closest_node for item in sublist]
SaveClusterAvg = [item for sublist in SaveClusterAvg for item in sublist]
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np
import math

# Assuming you have imported the necessary classes and functions

G = Join_Graphs.copy()
G1 = Join_Graphs.copy()

visitNode = []

def process_node(node, visited):
    # Check if the node has already been visited
    if node in visited:
        return

    # Mark the current node as visited
    visited.add(node)

    visitNode.append(node)

    degree = G.degree(node)
    print(f'Degree of {node}: {degree}')
    if degree > 3 and eval(node) in save_closest_node :
        neighbors = []
        outgoing_neighbors = list(G.successors(node))
        incoming_neighbors = list(G.predecessors(node))
        if len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 2:
            neighbors.append(outgoing_neighbors)
            neighbors.append(incoming_neighbors)
            neighbors = [item for sublist in neighbors for item in sublist]
            neighbors = [neighbor for neighbor in neighbors if neighbor not in map(str, SaveClusterAvg)]

        elif len(outgoing_neighbors) != 0 or len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 1:
            neighbors = list(G.neighbors(node))



        # neighbors = list(G.neighbors(node))

        # Create clusters based on neighbors
        clusters = cluster_neighbors(neighbors)

        if clusters:
            average_nodes, neighbor_lists = calculate_average_and_update_graph(clusters, G, node)
            # process_new_node(average_nodes, visited)
            # Process the newly created nodes and their neighbors
            for neighbor in neighbors:
                process_new_node(neighbor, visited)
    elif degree>3 and  eval(node) not in save_closest_node:
        neighbors = list(G.neighbors(node))
        clusters = cluster_neighbors(neighbors)

        if clusters:
            average_nodes, neighbor_lists = calculate_average_and_update_graph(clusters, G, node)
            # process_new_node(average_nodes, visited)
            # Process the newly created nodes and their neighbors
            for neighbor in neighbors:
                process_new_node(neighbor, visited)
def cluster_neighbors(neighbors):
    # Initialize a list to save clusters
    SaveCluster = []

    # Convert the neighbor coordinates to a NumPy array
    w = [list(map(float, element.strip('[]').split(','))) for element in neighbors]
    w_array = np.array(w)

    # Create a KMeans clustering model with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)

    # Fit the model to the data
    kmeans.fit(w_array)

    # Get cluster labels
    labels = kmeans.labels_

    # Separate the points into two clusters based on the labels
    cluster_1 = [w[i] for i in range(len(w)) if labels[i] == 0]
    cluster_2 = [w[i] for i in range(len(w)) if labels[i] == 1]

    # Append the clusters to SaveCluster
    SaveCluster.append(cluster_1)
    SaveCluster.append(cluster_2)

    return SaveCluster

def calculate_average_and_update_graph(SaveCluster, G, node):
    save_AVG = []  # Initialize a list to save average coordinates
    average_nodes = []  # Initialize a list to store average nodes
    neighbor_lists = []  # Initialize a list to store neighbors of average nodes

    for cluster_index, coordinates in enumerate(SaveCluster):
        if len(coordinates) > 1:
            coorNode = eval(node)
            # Calculate the average point
            average_coordinate = [
                (coorNode[0] + sum(point[0] for point in coordinates)) / (len(coordinates) + 1),
                (coorNode[1] + sum(point[1] for point in coordinates)) / (len(coordinates) + 1),
                (coorNode[2] + sum(point[2] for point in coordinates)) / (len(coordinates) + 1)
            ]

            # Print the average point
            print("Average Point:", average_coordinate)
            save_AVG.append(average_coordinate)

            # Add the average coordinate to the graph
            average_vertex = Vertex(average_coordinate)
            currnode = Vertex(eval(node))

            G.add_node(str(average_vertex.point), pos=average_vertex.point)
            G.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))
            Join_Graphs.add_node(str(average_vertex.point), pos=average_vertex.point)
            Join_Graphs.add_edge(str(currnode.point), str(average_vertex.point), object=Edge(currnode, average_vertex))

            Connect_edges = list(Join_Graphs.edges(node))
            sav_radius_clustering = []

            for coord in coordinates:
                key1 = str(coord)
                key2 = str(node)

                if key1 in Join_Graphs[key2] and 'radius' in Join_Graphs[key2][key1]:
                    sav_radius_clustering.append(Join_Graphs[key2][key1]['radius'])
                elif key2 in Join_Graphs[key1] and 'radius' in Join_Graphs[key1][key2]:
                    sav_radius_clustering.append(Join_Graphs[key1][key2]['radius'])
                else:
                    # Handle the case where 'radius' is not present
                    default_radius = 1
                    Join_Graphs[key2][key1] = {'radius': default_radius, 'diameter': 2}
                    sav_radius_clustering.append(default_radius)

            averageRad = sum(sav_radius_clustering) / len(sav_radius_clustering)
            Join_Graphs[node][str(average_coordinate)]['radius'] = averageRad
            Join_Graphs[node][str(average_coordinate)]['diameter'] = 2 * averageRad

            # Calculate the Euclidean distance between the node and the average point
            distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
            Join_Graphs[node][str(average_coordinate)]['length'] = distance

            print(f'Cluster {cluster_index + 1} - Average Coordinate: {average_vertex.point}')

            for t in range(len(coordinates)):
                node_SaveCluster = Vertex(coordinates[t])
                G.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
                Join_Graphs.add_edge(str(average_vertex.point), str(node_SaveCluster.point), object=Edge(average_vertex, node_SaveCluster))
                Join_Graphs[str(average_coordinate)][str(coordinates[t])]['radius'] = 0.4*averageRad
                Join_Graphs[str(average_coordinate)][str(coordinates[t])]['diameter'] = 2 * (0.4*averageRad)
                distance = euclidean_distance(tuple(eval(node)), tuple(average_coordinate))
                Join_Graphs[node][str(average_coordinate)]['length'] = distance

            # Save the average node and its neighbors for later processing
            average_nodes.append(average_vertex.point)
            neighbor_lists.append(coordinates)

            resulting_clusters_sav = coordinates
            connected_edges = list(G.edges(node))
            ##### to check if we can find all edges of the specific node############
            elements_in_connected_edges = []
            for edge in connected_edges:
                for e in range(len(resulting_clusters_sav)):
                    if str(resulting_clusters_sav[e]) in edge:
                        elements_in_connected_edges.append(resulting_clusters_sav[e])


            ########## this part for finding the missing edges for specific node ################
            # Convert the elements in resulting_clusters_sav to strings
            resulting_clusters_sav_str = [str(cluster) for cluster in resulting_clusters_sav]

            # Create the combined list without repetitions
            combined_list = connected_edges + list(set((
                (connected_edge[0], resulting_cluster)
                for connected_edge in connected_edges
                for resulting_cluster in resulting_clusters_sav_str# Convert elements to strings for comparison
            )))




            if len(elements_in_connected_edges) != 0:
                for edge in connected_edges:
                    for nbn in resulting_clusters_sav:
                        if str(nbn) in edge:
                            G.remove_edge(*edge)
                            Join_Graphs.remove_edge(*edge)

            else:
                for edge in combined_list:
                    for nbn in resulting_clusters_sav_str:
                         if str(nbn) in edge:
                            # Check if the edge or its reverse exists in the graph before removing
                            if G.has_edge(*edge):
                                G.remove_edge(*edge)
                            elif G.has_edge(edge[1], edge[0]):
                                G.remove_edge(edge[1], edge[0])
                            if Join_Graphs.has_edge(*edge):
                                Join_Graphs.remove_edge(*edge)
                            elif Join_Graphs.has_edge(edge[1], edge[0]):
                                Join_Graphs.remove_edge(edge[1], edge[0])









    return average_nodes, neighbor_lists

# Define the Euclidean distance function
def euclidean_distance(point1, point2):
    if len(point1) != 3 or len(point2) != 3:
        raise ValueError("Both points should be 3D coordinates.")

    x1, y1, z1 = point1
    x2, y2, z2 = point2

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance

# Define the process_new_node function
def process_new_node(node, visited, neighbors=None):
    if node in visited:
        return

    visited.add(node)

    degree = G.degree(node)
    print(f'Degree of {node}: {degree}')

    if degree > 3 and eval(node) in save_closest_node :
        neighbors1 = []
        outgoing_neighbors = list(G.successors(node))
        incoming_neighbors = list(G.predecessors(node))
        if len(outgoing_neighbors) != 0 and len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 2:
            neighbors1.append(outgoing_neighbors)
            neighbors1.append(incoming_neighbors)
            neighbors1 = [item for sublist in neighbors1 for item in sublist]
            neighbors1 = [neighbor for neighbor in neighbors1 if neighbor not in map(str, SaveClusterAvg)]

        elif len(outgoing_neighbors) != 0 or len(incoming_neighbors)!= 0 and len(outgoing_neighbors) <= degree - 1:
            neighbors1 = list(G.neighbors(node))

        resulting_clusters = cluster_neighbors(neighbors1)

        if resulting_clusters:
            # For example, you can use your existing code to create the average point and add it to the graph
            average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
            print(average_nodes)

            # After adding the new node and edges, you can process the neighbors
            if neighbors:
                for neighbor in neighbors:
                    process_new_node(neighbor, visited)

    elif degree>3 and  eval(node) not in save_closest_node:
        neighbors1 = list(G.neighbors(node))
        # if len(neighbors1) == degree - 1:


        resulting_clusters = cluster_neighbors(neighbors1)
        if resulting_clusters:
            # For example, you can use your existing code to create the average point and add it to the graph
            average_nodes, neighbor_lists = calculate_average_and_update_graph(resulting_clusters, G, node)
            print(average_nodes)

            # After adding the new node and edges, you can process the neighbors
            if neighbors:
                for neighbor in neighbors:
                    process_new_node(neighbor, visited)

# Create a set to keep track of visited nodes
visited_nodes = set()


while True:
    nodes_with_degree_greater_than_3 = [node for node in G.nodes if G.degree(node) > 3]
    if not nodes_with_degree_greater_than_3:
        break
    for node in nodes_with_degree_greater_than_3:
        process_node(node, visited_nodes)




############ recalculate the radius value in join graph###################################
for b in range(len(Roots)):
    Root_points = Roots[b]
    for rootN in Root_points:
        if Join_Graphs.has_node(str(rootN)):
# rootN = '[145.470187956702, 139.918452882767, 196.563]'
            bfs_tree = nx.bfs_tree(Join_Graphs, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)

            # Find the parent of each node
            parent = {}
            for node in bfs_tree.nodes():
                parent_nodes = list(bfs_tree.predecessors(node))
                if parent_nodes:
                    parent[node] = parent_nodes[0]
                else:
                    parent[node] = None


            for edge in bfs_tree.edges():
                parent[edge[1]] = edge[0]


            # Print the parents of each node
            print("Parents of each node:", parent)

            check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
            node_list = list(bfs_tree.nodes())
            edge_list = list(bfs_tree.edges())

            check = []
            ratiocheck = {}
            Diamcheck = False
            for node in bfs_tree.nodes():

                node = str(node)
                # node_index = node_list.index(node)
                node_index = list(bfs_tree.nodes()).index(node)
                if check_node[node_index] == False:

                    if parent[node] != None:

                            par = Join_Graphs[parent[node]][node]["radius"]

                            if len(list(Join_Graphs.neighbors(node))) > 0:
                                nnb = list(Join_Graphs.neighbors(node))

                                # if str(rootN) in nnb:
                                #     nnb.remove(str(rootN))
                                queue = []
                                for j in range(len(nnb)):
                                    nnb_index = list(bfs_tree.nodes()).index(nnb[j])
                                    if check_node[nnb_index] == False:
                                        if par!= 0:
                                            # children = nnb
                                            child = Join_Graphs[node][nnb[j]]["radius"]
                                            if child != 0:
                                                ratioDiam = (child / par)

                                                check_node[node_index] = True
                                                check.append(node)
                                                # node_index = node_list.index(nnb[j])

                                                # check_node[nnb_index] = True
                                                if ratioDiam >1:

                                                    Join_Graphs[node][nnb[j]]["radius"] = par - (0.4 * par)
                                                    Join_Graphs[node][nnb[j]]["diameter"] = 2*(par - (0.4 * par))

                                                    par = Join_Graphs[parent[node]][node]["diameter"]
                                                    child = Join_Graphs[node][nnb[j]]["diameter"]
                                                    ratioDiam = (child / par)
                                                    Join_Graphs[node][nnb[j]]["ratioDiam"] = ratioDiam
                                            else:
                                                Join_Graphs[node][nnb[j]]["radius"] = par - (0.4* par)
                                                Join_Graphs[node][nnb[j]]["diameter"] = 2*(par - (0.4 * par))
                                                child = Join_Graphs[node][nnb[j]]["radius"]
                                                ratioDiam = (child / par)

                                                check_node[node_index] = True
                                                check.append(node)
                                                # node_index = node_list.index(nnb[j])

                                                # check_node[nnb_index] = True
                                                if ratioDiam >1:

                                                    Join_Graphs[node][nnb[j]]["radius"] = par - (0.4 * par)
                                                    Join_Graphs[node][nnb[j]]["diameter"] = 2*(par - (0.4 * par))

                                                    par = Join_Graphs[parent[node]][node]["diameter"]
                                                    child = Join_Graphs[node][nnb[j]]["diameter"]
                                                    ratioDiam = (child / par)
                                                    Join_Graphs[node][nnb[j]]["ratioDiam"] = ratioDiam



def predict_side_branches_for_final_graph(graph, Roots, rad):
    g = graph.copy() #one copy for reading the nodes from the perivious graph
    g1 = graph.copy() # another copy for adding the new branches
    avglen = []
    avgratiolendiam = []
    for b in range(len(Roots)):
        Root_points = Roots[b]
        for rootN in Root_points:
            if Join_Graphs.has_node(str(rootN)):
                bfs_tree = nx.bfs_tree(Join_Graphs, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)
    
                # Find the parent of each node
                parent = {}
                for node in bfs_tree.nodes():
                    parent_nodes = list(bfs_tree.predecessors(node))
                    if parent_nodes:
                        parent[node] = parent_nodes[0]
                    else:
                        parent[node] = None
    
    
                for edge in bfs_tree.edges():
                    parent[edge[1]] = edge[0]
    
    
                # Print the parents of each node
                print("Parents of each node:", parent)
    
                check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
                node_list = list(bfs_tree.nodes())
                edge_list = list(bfs_tree.edges())
                
                
                for u, v, o in g.edges(data="object"):
                    if (u, v) in edge_list:
                        if g[u][v]["RatioLendiameter"] <= 5:
                             avgratiolendiam.append(g[u][v]["RatioLendiameter"])
                             avglen.append(g[u][v]["length"])
            
                avg_ratioLenDiam = statistics.mean(avgratiolendiam)
                avg_ratioLenDiam = math.ceil(avg_ratioLenDiam)
                avg_len = statistics.mean(avglen)
                avg_len = math.ceil(avg_len)

                for  u, v, o in g.edges(data="object"):
                    if (u, v) in edge_list:
                        if g[u][v]["RatioLendiameter"] > 5:



                            branches = [(np.array(ast.literal_eval(u)), np.array(ast.literal_eval(v))),]
            
                            g1.remove_edge(u, v)
                            # g.remove_node(u)
                            # g.remove_node(v)
            
            
                            newlen = (g[u][v]["length"] / avg_len)
                            # newlen = math.ceil(newlen) # round the float number to big
                            newlen = math.floor(newlen) - 2 # round the float number to small
                            divisions = newlen
            
                            # new_branches = []
                            for start_point, end_point in branches:
                                length = np.linalg.norm(end_point - start_point)
                                # direction_vector = (end_point - start_point) / length
                                # division_lengths = np.linspace(0, length, divisions + 1)
                                step = (end_point - start_point) / (divisions + 1)
                                new_nodes = [list(start_point)]
            
                                # for i in range(1, len(division_lengths)):
                                #     new_start_point = list(start_point + division_lengths[i-1] * direction_vector)
                                #     new_end_point = list(start_point + division_lengths[i] * direction_vector)
                                #     new_branches.append((new_start_point, new_end_point))
                                for i in range(1, divisions + 1):
                                    new_node = start_point + i * step
                                    new_nodes.append(list(new_node))
            
                                # Add the end node
                                new_nodes.append(list(end_point))
                                new_branches = []
            
                                for i in range(len(new_nodes) - 1):
                                    segment = (new_nodes[i], new_nodes[i+1])
                                    new_branches.append(segment)
            
            
            
                            # if not all(component >= 0 for component in vector[:2]):
                            #     new_branches = new_branches
                            #
                            # elif all(component >= 0 for component in vector[:2]):
                            new_branches.reverse()
                            new_branches = [tuple(reversed(item)) for item in new_branches]
            
                            for w in range(len(new_branches)):
                                add_more_newbranches = []
                                if len(new_branches) > 0:
                                    if not str((new_branches[w][0])) in g1.nodes() or not str((new_branches[w][1])) in g1.nodes():
                                        if not str((new_branches[w][0])) in g1.nodes():
                                            startpoint = Vertex(((new_branches[w][0])))
                                            endpoint = Vertex(((new_branches[w][1])))
                                            g1.add_node(str(startpoint.point), pos=startpoint.point)
                                            g1.add_node(str(endpoint.point), pos=endpoint.point)
                                            g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
                                        elif not str((new_branches[w][1])) in g1.nodes():
                                            startpoint = Vertex(((new_branches[w][1])))
                                            endpoint = Vertex(((new_branches[w][0])))
                                            g1.add_node(str(startpoint.point), pos=startpoint.point)
                                            g1.add_node(str(endpoint.point), pos=endpoint.point)
                                            g1.add_edge(str(startpoint.point), str(endpoint.point), object=Edge(startpoint, endpoint))
            
            
                                            new_branch_start = startpoint.point
                                            main_endpoint = endpoint.point
            
                                            g1[str(new_branch_start)][str(main_endpoint)]['length'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))
                                            g1[str(new_branch_start)][str(main_endpoint)]['radius'] = (g[u][v]['radius'])
                                            g1[str(new_branch_start)][str(main_endpoint)]['diameter'] = 2*(g[u][v]['radius'])
                                            g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'] = np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start)) /  2*(g[u][v]['radius'])
                                            # g1[str(new_branch_start)][str(main_endpoint)]['weight'] = -1
            
            
            
                                            if str(new_branch_start) not in rad:
                                                rad[str(new_branch_start)] = (g[u][v]['radius'])
            
                                            if str(main_endpoint) not in rad:
                                                rad[str(main_endpoint)] = (g[u][v]['radius'])
            
            
            
                                            # direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                            # vector = direction_vector
                                            # if not all(component <= 0 for component in vector):
            
                                            if w % 2 == 0:
            
            
                                                direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                                normalized_direction = direction_vector / np.linalg.norm(direction_vector)
            
                                                # Calculate the angle in radians (40 degrees)
                                                angle_degrees = 40
                                                angle_radians = np.radians(angle_degrees)
            
                                                rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                                [0, 1, 0],
                                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
            
                                                # Rotate the direction vector to get the new direction
                                                new_direction = np.dot(rotation_matrix, normalized_direction)
            
            
                                                # Calculate the length of the new branch (half of the main branch)
                                                new_branch_length = 0.5 * np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))
            
                                                # Calculate the new end node's coordinates
                                                new_end = np.array(new_branch_start) + new_branch_length * new_direction
            
                                                print(f"New end node coordinates: {new_end}")
            
            
            
                                                newpoint = Vertex((list(new_end)))
                                                g1.add_node(str(newpoint.point), pos=newpoint.point)
                                                g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))
            
            
                                                newpoint_end = newpoint.point
                                                print(newpoint_end)
                                                g1[str(new_branch_start)][str(newpoint_end)]['radius'] = 0.4 * (g[u][v]['radius'])
                                                g1[str(new_branch_start)][str(newpoint_end)]['length'] = new_branch_length
                                                g1[str(new_branch_start)][str(newpoint_end)]['diameter'] = 2* (0.4 * (g[u][v]['radius']))
                                                g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'] = new_branch_length /  2* (0.4 * (g[u][v]['radius']))
                                                # g1[str(startpoint.point)][str(newpoint.point)]['weight'] = -1
                                                rad[str(newpoint.point)] = (0.4 * (g[u][v]['radius']))
            
            
                                                # g1[startpoint.point][newpoint.point]['length'] = half_main_branch_length
            
                                                # Print the new branch coordinates
                                                print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                                                # network_plot_3D(g, angle=0)
                                            else:
                                                direction_vector = np.array(main_endpoint) - np.array(new_branch_start)
                                                normalized_direction = direction_vector / np.linalg.norm(direction_vector)
            
                                                # Calculate the angle in radians (40 degrees)
                                                angle_degrees = -40
                                                angle_radians = np.radians(angle_degrees)
            
                                                rotation_matrix = np.array([[np.cos(angle_radians), 0, np.sin(angle_radians)],
                                                [0, 1, 0],
                                                [-np.sin(angle_radians), 0, np.cos(angle_radians)]])
            
                                                # Rotate the direction vector to get the new direction
                                                new_direction = np.dot(rotation_matrix, normalized_direction)
            
            
                                                # Calculate the length of the new branch (half of the main branch)
                                                new_branch_length = 0.5 * np.linalg.norm(np.array(main_endpoint) - np.array(new_branch_start))
            
                                                # Calculate the new end node's coordinates
                                                new_end = np.array(new_branch_start) + new_branch_length * new_direction
            
                                                # print(f"New end node coordinates: {new_end}")
            
            
            
                                                newpoint = Vertex((list(new_end)))
                                                g1.add_node(str(newpoint.point), pos=newpoint.point)
                                                g1.add_edge(str(startpoint.point), str(newpoint.point), object=Edge(startpoint, newpoint))
            
                                                newpoint_end = newpoint.point
                                                print(newpoint_end)
                                                g1[str(new_branch_start)][str(newpoint_end)]['radius'] = 0.4 * (g[u][v]['radius'])
                                                g1[str(new_branch_start)][str(newpoint_end)]['length'] = new_branch_length
                                                g1[str(new_branch_start)][str(newpoint_end)]['diameter'] = 2*(0.4 * (g[u][v]['radius']))
                                                g1[str(new_branch_start)][str(newpoint_end)]['RatioLendiameter'] = new_branch_length /  2* (0.4 * (g[u][v]['radius']))
                                                # g1[str(startpoint.point)][str(newpoint.point)]['weight'] = -1
                                                rad[str(newpoint.point)] = (0.4 * (g[u][v]['radius']))
            
                                                # g1[startpoint.point][newpoint.point]['length'] = half_main_branch_length
            
                                                # Print the new branch coordinates
                                                print(f"New Branch - Start: {new_branch_start}, End: {newpoint.point}")
                                                # network_plot_3D(g, angle=0)
            
                                    elif str((new_branches[w][0])) in g1.nodes() and str((new_branches[w][1])) in g1.nodes():
                                         new_branch_start = Vertex(new_branches[w][0])
                                         main_endpoint = Vertex(new_branches[w][1])
                                         g1.add_node(str(new_branch_start.point), pos=new_branch_start.point)
                                         g1.add_node(str(main_endpoint.point), pos=main_endpoint.point)
                                         g1.add_edge(str(new_branch_start.point), str(main_endpoint.point), object=Edge(new_branch_start, main_endpoint))
            
            
                                         g1[str(new_branch_start)][str(main_endpoint)]['length'] = np.linalg.norm(np.array(main_endpoint.point) - np.array(new_branch_start.point))
                                         g1[str(new_branch_start)][str(main_endpoint)]['radius'] = (g[u][v]['radius'])
                                         g1[str(new_branch_start)][str(main_endpoint)]['diameter'] = 2 * (g[u][v]['radius'])
                                         g1[str(new_branch_start)][str(main_endpoint)]['RatioLendiameter'] = np.linalg.norm(np.array(main_endpoint.point) - np.array(new_branch_start.point)) / 2 * (g[u][v]['radius'])
                                         # g1[str(new_branch_start)][str(main_endpoint)]['weight'] = -1
            
            
            
            
                return g1, rad
                            
            

graph_update, new_dic_new = predict_side_branches_for_final_graph(Join_Graphs, Roots, new_dic)            
            
Final_graph_update, RootBranches2 = Final_graph_Group2(graph_withsidebranches_group2, new_dic_new)





    

    
    


    
    
    
############################### remove small and wrong trees in graph ###################################
# Find weakly connected components
wccs = list(nx.weakly_connected_components(Join_Graphs))

if len(wccs) >=1:
    # Set a threshold for the minimum size of a connected component
    min_component_size = 20  # Set your desired threshold

    # Identify components to remove
    components_to_remove = [component for component in wccs if len(component) <= min_component_size]

    # Remove identified components
    for component in components_to_remove:
        for node in component:
            Join_Graphs.remove_node(node)





#
# rootN = '[110.004, 160.894, 197.801]'
# bfs_tree = nx.bfs_tree(Join_Graphs, str(rootN), reverse=False, depth_limit=None, sort_neighbors=True)
# parent = {}
# for node in bfs_tree.nodes():
#     parent_nodes = list(bfs_tree.predecessors(node))
#     if parent_nodes:
#         parent[node] = parent_nodes[0]
#     else:
#         parent[node] = None
#
# check_node = [False for x in range(1) for y in range(len(bfs_tree.nodes()))]
# node_list = list(bfs_tree.nodes())
# edge_list = list(bfs_tree.edges())
#
# check = []
# ratiocheck = {}
# Diamcheck = False
# for node in bfs_tree.nodes():
#
#     node = str(node)
#     # node_index = node_list.index(node)
#     node_index = list(bfs_tree.nodes()).index(node)
#     if check_node[node_index] == False:
#
#         if parent[node] != None:
#
#                 # par = Join_Graphs[parent[node]][node]["radius"]
#
#                 if len(list(Join_Graphs.neighbors(node))) > 0:
#                     nnb = list(Join_Graphs.neighbors(node))
#
#                     # if str(rootN) in nnb:
#                     #     nnb.remove(str(rootN))
#                     queue = []
#                     for j in range(len(nnb)):
#                         nnb_index = list(bfs_tree.nodes()).index(nnb[j])
#                         if check_node[nnb_index] == False:
#
#                             a = np.array(ast.literal_eval(node)) - np.array(ast.literal_eval(parent[node]))
#                             b = np.array(ast.literal_eval(nnb[j])) - np.array(ast.literal_eval(node))
#                             inner = np.inner(a, b)
#                             norms = LA.norm(a) * LA.norm(b)
#                             cos = inner / norms
#                             radianc = np.arccos(np.clip(cos, -1.0, 1.0))
#                             # rad = math.acos(cos)
#                             angle = np.rad2deg(radianc)
#                             # Set a threshold angle (adjust as needed)
#                             angle_threshold = 90.0
#
#                             # Check if the angle is greater than the threshold
#                             if angle < angle_threshold:
#                                 branch = [node, parent[node], nnb[j]]
#                                 # Remove the branch (adjust based on your graph representation)
#                                 Join_Graphs.remove_nodes_from(branch)
#










#
# # Specify the start node
# start_node = '[158.011, 150.247, 191.52]'
# # Find parents for each node in the graph and store the results in a dictionary
# parents_dict = {node: list(Join_Graphs.predecessors(node)) for node in Join_Graphs.nodes}
# # Display the parents for each node
# for node, parents in parents_dict.items():
#     print(f"Node: {node}, Parents: {parents}")
#
# visited = set()
# startN  = '[158.011, 153.247, 201.52]'
# for n in Join_Graphs.nodes:
#     visited.add(str(n))
#     if str(n) != startN and parents_dict[str(n)] == []:
#              edges_connected_to_node = list(Join_Graphs.edges(str(n)))
#              for i in range(len(edges_connected_to_node)):
#                     for j in range(len(edges_connected_to_node[i])):
#                            if str(edges_connected_to_node[i][j]) != str(n) and str(edges_connected_to_node[i][j]) in visited:
#                                 edge_to_reverse = edges_connected_to_node[i]
#
#                                 # Reverse the direction of the edge
#
#                                 reversed_edge = (edge_to_reverse[1], edge_to_reverse[0])
#
#                                 # Update the graph with the reversed edge
#                                 Join_Graphs.remove_edge(*edge_to_reverse)
#                                 Join_Graphs.add_edge(*reversed_edge)
#
# visited = set()
# startN  = '[158.011, 153.247, 201.52]'
# for n1 in Join_Graphs.nodes:
#     visited.add(str(n1))
#     if str(n1) != startN and parents_dict[str(n1)] == []:
#          edges_connected_to_node = list(Join_Graphs.edges(str(n1)))
#          for i in range(len(edges_connected_to_node)):
#                 for j in range(len(edges_connected_to_node[i])):
#                        if str(edges_connected_to_node[i][j]) != str(n1) :
#                             edges = [edges_connected_to_node[i]]
#                             for edge in edges:
#                                 start_point = eval(edge[0])
#                                 end_point = eval(edge[1])
#                                 slope = [end_point[i] - start_point[i] for i in range(len(start_point))]
#                                 if all(s > 0 for s in slope):
#                                     print(f"Slope for {edge}: {slope}, All elements are positive.")
#                                     edge_to_reverse = edge
#                                     print(edge_to_reverse)
#
#                                     # Reverse the direction of the edge
#
#                                     reversed_edge = (edge_to_reverse[1], edge_to_reverse[0])
#
#                                     # Update the graph with the reversed edge
#                                     Join_Graphs.remove_edge(*edge_to_reverse)
#                                     Join_Graphs.add_edge(*reversed_edge)





writeGraph(Join_Graphs)
