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

# mesh = pyvista.Sphere()
# df = pd.read_csv('test-nb-pyvista.csv', header=None)
# coor = []
# for index, row in df.iterrows():
#     coor.append([row[0], row[1], row[2]])
#
# coor = np.array(coor)
# point = coor[0]
# tree = KDTree(X, leaf_size=X.shape[0]+1)
# distances, ind = tree.query([point], k=2)
# print(X[ind])

# for i in range(len(coor)):
#     point = coor[i]
#     tree = KDTree(X, leaf_size=X.shape[0] + 1)
#
#     distances, ind = tree.query([point], k=2)
#     print(distances)
#     print(ind)
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
def buildTree_trunk(coor, start=None):
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
    while nWhitePixels != blackedPixels:


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


            # point = b[i]

            # queue = collections.deque()
            # for h in range(len(point_array)):
            #     queue.append(point_array[h])
            if new_start:
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

            # if distances[0][1] > 8: # seprate the small trees
            #     blackedPixels += 1
            #     coor.remove(point)
            #     break

            for j in range(1, len(distances[0])):

                if len(delpoint)!=0:
                    dic = {}
                    for c in range(len(delpoint)):
                        dist = np.linalg.norm(np.array(point) - np.array(delpoint[c]))
                        dic[dist] = delpoint[c]
                    per_point = dic[min(dic.keys())]

                    if not G.has_edge(str(point), str(per_point)) and min(dic.keys()) < 1.8:  #1.5033
                        currV = Vertex(point)
                        pervious_point = Vertex(per_point)
                        G.add_node(str(currV.point), pos=currV.point)
                        G.add_node(str(pervious_point.point), pos=pervious_point.point)
                        G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))


                #X.append(point)
                if distances[0][j] <= 0.625:

                    nb_closest = coor[ind[0][j]]
                    if nb_closest not in X:
                        X.append(nb_closest)

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
                        if not G.has_edge(str(point),str(coor[ind[0][j]])):
                            currV = Vertex(point)
                            pervious_point = Vertex(coor[ind[0][j]])
                            G.add_node(str(currV.point), pos=currV.point)
                            G.add_node(str(pervious_point.point), pos=pervious_point.point)
                            G.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))
                    # coor.remove(currV.point)
                            #network_plot_3D(G, angle=0)
                    # point_array = np.delete(point_array, i, axis=0)

            if count_neigh == 0 and distances[0][1]<8:
                disconect_tree = False
                #network_plot_3D(G, angle=0)

                for j in range(1, len(distances[0])):
                    nb_closest = coor[ind[0][j]]
                    if nb_closest not in X:
                        X.append(nb_closest)
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
                        if not G.has_edge(str(point),str(coor[ind[0][j]])):
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
                        if not G.has_edge(str(currV.point),str(nb_closest_Vertex.point)) and nb_closest_Vertex.point > currV.point:
                            X.append(nb_closest)
                            newV = Vertex(nb_closest)

                            currV = Vertex(point)
                            G.add_node(str(currV.point), pos=currV.point)
                            G.add_node(str(newV.point), pos=newV.point)
                            G.add_edge(str(currV.point), str(newV.point), object=Edge(currV, newV))

                            #blackedPixels += 1
                            #network_plot_3D(G, angle=0)
                            break
            elif count_neigh == 0 and distances[0][1]>8: #remove the edge from ends points
                i = i + 1
                blackedPixels += 1
                coor.remove(point)
                break


            i = i + 1
            blackedPixels += 1

            if len(coor) > 1:
                delpoint.append(currV.point)
                coor.remove(currV.point) # remove the currv from the list and find the other neigbours
            else:
                break





            # queue = collections.deque()
            # queue.append(point_array)
            # queue.popleft()
            # p = np.array(queue)

        # X.popleft()

        graphs.append(G)



        # empty queue
        # current graph is finished ->store it
        #graphs.append(G)
        G_composed = nx.compose_all(graphs)

        # network_plot_3D(G,0)
        # display(G)


        # reset start
        start = None
        # H = nx.compose(H, G)

    # end while
    # mlab.show()

    return G


def buildTree_all_branch(coor1, coor, trimmedGraph, endNodes_main_artery, start=None):
    X = []

    a=[]
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



    # network_plot_3D(g, 0)
    # display(g)
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
                if g[p][q]["weight"] > dist and dist <10:
                    g.remove_node(p)
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                elif g[p][q]["weight"] < 10:
                    g.remove_node(p)
                    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!removed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
            if g[v0][x0]["weight"] < 3:  # think honiiiiiiiiiiiiii
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
    else:
        avg_radius = 0
    crossSecArea = pi * avg_radius ** 2
    return crossSecArea,avg_radius

def avgBranchRadius (combinedGraph,new_img_dst,img):  #new_img_dst
    """Input: Built graph, distance transform of vesselmask
    First calculates additional distances of ending nodes from its background and adds it to the
    branch length. Next, for longer branches with distance > 5, pick intermediate pixels and calculate
    radial distance. Add length, radius, cross-sectional area to graph dataset. Repeat for smaller edges <= 5,
    only difference is that  intermediate pixels are only 3 pixels of the middle ones."""

    # get all ending nodes
    for u, v, o in combinedGraph.edges(data="object"):

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
            csaAvg, radiusAvg = radialDistance(betweenPxlList,radii)  #new_img_dst
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
            combinedGraph[u][v]["csa"] = csaAvg
            # combinedGraph[u][v]["volume"] = volume

            ###############################################################
            # avgVolume = segmentedBranchVolume(pixelList=o.pixels,new_img_dst=new_img_dst,endNodeAddition=endNodeAddition)
            # combinedGraph[u][v]["csa"] = csaAvg
            # combinedGraph[u][v]["volume"] = avgVolume
            ###################################################################

        elif weight < 5 and weight >= 2:
            mid = weight//2
            csaSmall,radiusSmall = radialDistance(o.pixels[mid-1:mid+1],new_img_dst)  #new_img_dst
            # volumeS = csaSmall * (pixelDist + endNodeAddition)
            combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition
            combinedGraph[u][v]["radius"] = radiusSmall
            combinedGraph[u][v]["csa"] = csaSmall
            # combinedGraph[u][v]["volume"] = volumeS
            # print("smaller branch","CSA", csaSmall,"weight",weight,"volume",csaSmall*(weight + endNodeAddition))
        else:
            combinedGraph[u][v]["length"] = 0
            combinedGraph[u][v]["radius"] = 0
            combinedGraph[u][v]["csa"] = 0
            # combinedGraph[u][v]["volume"] = 0
            print("0 branch weight", weight)
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


    nx.write_edgelist(G, "/home/arah607/Desktop/outputGraph/lung001_UA.csv",  delimiter=' ') #, data=['length', 'radius'] #tree_CIP_right_13feb



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
#######################RUN#######################################################




# df = pd.read_csv('test-nb-pyvista.csv', header=None)

# df = pd.read_csv('small_tree_CIP_coords_55branches.csv', header=None)
# df = pd.read_csv('small_tree_CIP_coords_6branches.csv', header=None)
# df = pd.read_csv('small_tree_CIP_coords_5branches_radii.csv', header=None)
# df = pd.read_csv('two_small_tree_CIP.csv', header=None)
# df = pd.read_csv('small_lung (copy).csv', header=None)
# df1 = pd.read_csv('mpa_cip.csv', header=None)
# df = pd.read_csv('right_lung_artery_onebranch.csv', header=None)
# df = pd.read_csv('right_mainartery_CIP.csv', header=None)
# df1 = pd.read_csv('right_onesmall_branch_new.csv', header=None)
# df1 = pd.read_csv('test_branch_abovelope.csv', header=None)
# df1 = pd.read_csv('right_lung_artery_newwww.csv', header=None)
# df1 = pd.read_csv('small_upper_branch.csv', header=None)
# df1 = pd.read_csv('top_branch_9Feb.csv', header=None)
# df1 = pd.read_csv('3d_digit_creatingpoints.csv', header=None)
# df1 = pd.read_csv('trunk_mainpul_points.csv', header=None)
# df1 = pd.read_csv('onebranch_trunkmain_points.csv', header=None)
# df1 = pd.read_csv('modify_trunk_mainpul_points.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Graph_analysis/Vessel_Quantification/modify_trunk_mainpul_points_27April.csv', header=None)
# df1 = pd.read_csv('/home/arah607/Graph_analysis/Vessel_Quantification/trunknodes_mainartery_19020F.csv', header=None)

# value = df1.loc[0, "cohort"] # if we want see especific row and column with index row and string column

df1 = pd.read_csv('/home/arah607/Desktop/lung001_UpperArtery_Ben.csv', header=None) # the translation data for 15814w



df1 = df1.sort_values(0) # sort decending csv file, read main artery
# df.sort_values(df.columns[0, 1, 2], axis=0, inplace=True) # sort from high to low
coor = []
radii={}
for index, row in df1.iterrows():
    coor.append([row[0], row[1], row[2]])
    # radii[str([row[0], row[1], row[2]])] = row[3]




# df3= pd.read_csv('whole_right_lung_radiusfilter_translation.csv', header=None) # read other coor point with filter radius
# df3= pd.read_csv('whole_right_lung_radiusfilter_translation (copy).csv', header=None) # read other coor point with filter radius
# df3= pd.read_csv('top_filter_14feb.csv', header=None) # read other coor point with filter radius
# df3= pd.read_csv('allnodes_filter_radiibigger1.5_translation.csv', header=None) # read other coor point with filter radius
df3= pd.read_csv('/home/arah607/Graph_analysis/Vessel_Quantification/trunk_CIPdata_1may.csv', header=None) # example for some few branches
# df3= pd.read_csv('/home/arah607/Graph_analysis/Vessel_Quantification/test_one_branch.csv', header=None) # example for some few branches

df3 = df3.sort_values(0)
coor1 = []
radii1 = {}
for index, row in df3.iterrows():
    coor1.append([row[0], row[1], row[2]])
    # radii1[str([row[0], row[1], row[2]])] = row[3]



df2 = pd.read_csv('/home/arah607/Graph_analysis/Vessel_Quantification/only_pulmonary _trunk.csv', header=None) # read the pulmonary trunk
G1 = nx.Graph()
mpa_coor = []
for index, row in df2.iterrows():
    mpa_coor.append([row[0], row[1], row[2]])
    # coor.append((row[0], row[1], row[2]))
    mpa_coor.append([row[3], row[4], row[5]])
    # coor.append((row[3], row[4], row[5]))
    N1 = [row[0], row[1], row[2]]
    N2 = [row[3], row[4], row[5]]
    G1.add_node(str(N1), pos=N1)
    G1.add_node(str(N2), pos=N2)
    G1.add_edge(str(N1), str(N2), object=Edge(N1, N2))
endNodes_trunk = getEndNodes(G1) #save the all end points from trunk graph
from ast import literal_eval
for i in range(len(endNodes_trunk)):
    endNodes_trunk[i] = literal_eval(endNodes_trunk[i])
    coor1.append(endNodes_trunk[i])

############################################# read CIP vtk file ##############################################################
# dir = '/eresearch/lung/arah607/COPDgene/CIPtest/Normals/19003F/COPD1/19003F_INSP_B31f_340_COPD1'
#
# file = 'CTparticles.vtk_rightLungVesselParticles.vtk'
#
# mesh = pv.read(dir + '/' + file)
#
# coor = []
# for i in range(len(mesh.points)):
#     coor.append(mesh.points[i])
#
# coor = np.array(coor)
# coor = list(coor)
# for j in range(len(coor)):
#     coor[j] = list(coor[j])


# Connceted_Tree('tree_CIP_right_13feb.csv', 'connected_tree.csv')

H = nx.Graph()
# combined_graph = connected_component_to_graph(coor,H)
# coor.sort(key=lambda x: x[0]) #sort coor with trunk coordinates
# add the all end points from pulmonary trunk graph to coordinates of the inside lung
G = buildTree_trunk(coor, start=None)
network_plot_3D(G, angle=0)
writeGraph(G)
# writeGraph(G)
# mergedGraph = mergeEdges(G)
# mergedGraph = mergedGraph.copy()
# for index, row in df2.iterrows():
#     N1 = [row[0], row[1], row[2]]
#     N2 = [row[3], row[4], row[5]]
#     mergedGraph.add_node(str(N1), pos=N1)
#     mergedGraph.add_node(str(N2), pos=N2)
#     mergedGraph.add_edge(str(N1), str(N2), object=Edge(N1, N2))
# network_plot_3D(mergedGraph, angle=0)
# trimmedGraph = removeSmallEdge(mergedGraph)
# network_plot_3D(trimmedGraph, angle=0)
endNodes_main_artery = getEndNodes(G)


coor = []
for index, row in df1.iterrows():
    coor.append([row[0], row[1], row[2]])
GRAPH = buildTree_all_branches(coor1, coor, G, endNodes_main_artery, start=None)
network_plot_3D(GRAPH, angle=0)



####################################################### this part is not ready yet #####################################################
# mergedGraph1 = mergeEdges(GRAPH)
endNodes_main_artery11 = getEndNodes(GRAPH)
end11 = [] # all graph except main
from ast import literal_eval
for l in range(len(endNodes_main_artery11)):
    endNodes_main_artery11[l] = literal_eval(endNodes_main_artery11[l])
    end11.append(endNodes_main_artery11[l])

# end = [] # main pul artery graph
# for l in range(len(endNodes_main_artery)):
#     endNodes_main_artery[l] = literal_eval(endNodes_main_artery[l])
#     end.append(endNodes_main_artery[l])

dic = {}
# while len(endNodes_main_artery):
for i in range(len(endNodes_main_artery)):
    # per_point = []
    dic = {}
    for j in range(len(end11)):
        dist = np.linalg.norm(np.array(endNodes_main_artery[i]) - np.array(end11[j]))
        dic[dist] = end11[j]
    per_point = dic[min(dic.keys())]
    if not G.has_edge(str(endNodes_main_artery[i]), str(per_point)) and min(dic.keys()) < 5 and min(dic.keys()) != 0:
        currV = Vertex(endNodes_main_artery[i])
        pervious_point = Vertex(per_point)
        GRAPH.add_node(str(currV.point), pos=currV.point)
        GRAPH.add_node(str(pervious_point.point), pos=pervious_point.point)
        GRAPH.add_edge(str(currV.point), str(pervious_point.point), object=Edge(currV, pervious_point))

mergedGraph1 = mergeEdges(GRAPH)
# trimmedGraph1 = removeSmallEdge(mergedGraph1)
network_plot_3D(mergedGraph1, angle=0)

writeGraph(GRAPH)
Connceted_Tree(endNodes_main_artery, 'graph.csv', 'connected_tree.csv')  #tree_CIP_right_13feb

G_right = buildTree_all_branch(coor1, coor, trimmedGraph, endNodes_main_artery, start=None)
network_plot_3D(G_right, angle=0)


mergedGraph1 = mergeEdges(G_right)
network_plot_3D(mergedGraph1, angle=0)
# trimmedGraph1 = removeSmallEdge(mergedGraph1)
# network_plot_3D(trimmedGraph1, angle=0)
# remergedGraph1 = mergeEdges(trimmedGraph1)
# network_plot_3D(remergedGraph1, angle=0)

radii1.update(radii)
newCombinedGraph = avgBranchRadius(mergedGraph1, radii1, coor)

writeGraph(G)



# import numpy as np
# from scipy.spatial import KDTree
# import networkx as nx
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# centerline_coords = coor
# # Define function to create 3D graph
# def create_3d_graph(centerline_coords, r=5):
#     # Initialize empty graph
#     G = nx.Graph()
#
#     # Create KDTree to find nearest neighbors
#     kdtree = KDTree(centerline_coords)
#
#     # Add nodes to graph with coordinates as attributes
#     for i, coord in enumerate(centerline_coords):
#         G.add_node(i, coord=coord)
#
#     # Add edges between nodes with distance less than r
#     for i, coord in enumerate(centerline_coords):
#         dist, indices = kdtree.query(coord, k=10)
#         for j in indices[1:]:
#             if dist[j] <= r:
#                 G.add_edge(i, j, weight=dist[j])
#
#     # Return graph object
#     return G
#
#
# # Generate random centerline point coordinates
# np.random.seed(0)
# centerline_coords = np.random.rand(50, 3)
#
# # Create 3D graph with r=0.3
# G = create_3d_graph(centerline_coords, r=0.3)
#
# # Get node coordinates and draw 3D graph
# node_coords = nx.get_node_attributes(G, 'coord')
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# for i, coord in node_coords.items():
#     ax.scatter(coord[0], coord[1], coord[2], c='b', s=50)
# for u, v, d in G.edges(data=True):
#     ax.plot([node_coords[u][0], node_coords[v][0]], [node_coords[u][1], node_coords[v][1]],
#             [node_coords[u][2], node_coords[v][2]], 'k-', alpha=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
