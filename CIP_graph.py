import VesselVolumeCalc
import vessel_extraction
import vtk_nifit_render
import networkx as nx
import numpy as np
import scipy.ndimage
import skimage.measure
import visvis as vv
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from skimage.morphology import ball,cube
import glob
from PIL import Image
import scipy.ndimage
import cv2
from csv import writer
import placentagen as pg
from skimage import measure, morphology
from skan import skeleton_to_csgraph
from PIL import Image, TiffImagePlugin
from osgeo import gdal

###########################################################
"""
#Author: Joyce John
#This subroutine reads in vessel mask and centerline, builds graphs using the centerline, calculates
#cross section area of the vessel branches and volume of the vessel branch
"""
import pandas as pd
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from math import pi
from scipy import stats
import vtk_nifit_render
import visvis as vv
import skimage.io
import statistics
from skimage.morphology import ball,cube
import collections
import networkx as nx
import more_itertools
import math
import operator

########################################
import math

import time

import statistics

import numpy as np

from numpy.linalg import norm

import os

import pymesh

import pyvista as pv




def get_neighbours(p, exclude_p=False, shape=None):
    """subroutine to get 3x3x3 connected neighbour offsets"""

    ndim = len(p)

    # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)

    # offsets = np.r_[-3, 0, 3].take(offset_idx)


    # optional: exclude offsets of 0, 0, ..., 0 (i.e. p itself)
    if exclude_p:
        offsets = offsets[np.any(offsets, 1)]

    neighbours = p + offsets    # apply offsets to p

    # optional: exclude out-of-bounds indices
    # if shape is not None:
    #     valid = np.all((neighbours < np.array(shape)) & (neighbours >= 0), axis=1)
    #     neighbours = neighbours[valid]
    # print (offsets)
    return offsets

def buildTree(img, start=None):
    """Builds graph from centerline based on pixel connectivity. Gets every white pixel
    and converts it to black pixels once its added to the graph. Nodes are created using
    class Vertex and added to graph, similarly edges are created and added"""
    # copy image since we set visited pixels to black
    img = img.copy()
    img[np.nonzero(img)] = 1

    shape = img.shape
    nWhitePixels = np.sum(img)

    # to get offsets of surrounding voxels
    p = [10, 10, 10]
    offsets = get_neighbours(p)

    # neighbor offsets (8 nbors)
    # nbPxOff = np.array([[-1, -1], [-1, 0], [-1, 1],
    #                     [0, -1], [0, 1],
    #                     [1, -1], [1, 0], [1, 1]
    #                     ])
    nbPxOff = offsets
    queue = collections.deque()

    # a list of all graphs extracted from the skeleton
    graphs = []


    blackedPixels = 0
    # we build our graph as long as we have not blacked all white pixels!
    while nWhitePixels != blackedPixels:

        # if start not given: determine the first white pixel
        # if start is None:
        #     it = np.nditer(img, flags=['multi_index'])
        #     while not it[0]:
        #         it.iternext()
        #
        #     start = it.multi_index
        if start is None:
            # if not np.nonzero(img):
            #     print "all pixels blacked"
            #     break;
            start = np.transpose(np.nonzero(img))[0]
            print (type(start))
            # it = np.nditer(img, flags=['multi_index'])
            # while not it[0]:
            #     it.iternext()

            # start = it.multi_index

        startV = Vertex(start)
        queue.append(startV)
        # print("Start vertex: ", startV.point)

        # set start pixel to False (visited)
        img[startV.point[0], startV.point[1], startV.point[2]] = False
        blackedPixels += 1

        # create a new graph
        G = nx.Graph()

        G.add_node(startV,pos=startV.point)
        # G.add_node(startV, x=startV.point[0], y=startV.point[1], z=startV.point[2])

        # build graph in a breath-first manner by adding
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue):
            currV = queue[0]  # get current vertex
            # print("Current vertex: ", currV.point)

            # check all neigboor pixels
            for nbOff in nbPxOff:

                # pixel index

                # print currV.point
                pxIdx = currV.point + nbOff

                # print currV.point,pxIdx

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]) \
                        or (pxIdx[2] < 0 or pxIdx[2] >= shape[2]):
                    continue  # current neigbor pixel out of image

                if img[pxIdx[0], pxIdx[1], pxIdx[2]]:
                    # print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex([pxIdx[0], pxIdx[1], pxIdx[2]])


                    # add edge from currV <-> newV
                    G.add_edge(currV, newV, object=Edge(currV, newV))
                    # G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV, pos=newV.point)
                    # G.add_node(newV, x = startV.point[0], y = startV.point[1], z = startV.point[2])


                    # push vertex to queue
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0], pxIdx[1], pxIdx[2]] = False
                    blackedPixels += 1

            # pop currV
            queue.popleft()
        # end while

        # empty queue
        # current graph is finished ->store it
        graphs.append(G)
        G_composed = nx.compose_all(graphs)

        # network_plot_3D(G,0)
        # display(G)


        # reset start
        start = None
        # H = nx.compose(H, G)

    # end while
    # mlab.show()

    return graphs, G

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

def mergeEdges(graph):
    """
    # v0 -----edge 0--- v1 ----edge 1---- v2
    #        pxL0=[]       pxL1=[]           the pixel lists
    #
    # becomes:
    #
    # v0 -----edge 0--- v1 ----edge 1---- v2
    # v0 to v2
    #               new edge
    #    pxL = pxL0 + [v.point]  + pxL1      the resulting pixel list on the edge
    #
    # an delete the middle one
    # result:
    #
    # v0 --------- new edge ------------ v2
    #
    # where new edge contains all pixels in between!
    """

    # copy the graph
    g = graph.copy()

    # start not at degree 2 nodes
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]

    nb = []
    for v0 in startNodes:

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
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1))

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
    # print(nb)
    # weight the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)



    # network_plot_3D(g, 0)
    # display(g)
    return g

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
            if g[v0][x0]["weight"] < 5:  # think honiiiiiiiiiiiiii
                    # print("chosen weight",g[v0][x0]["weight"])
                g.remove_node(x0)
            # i=i+1

    return g

def getEndNodes(g):
    """Returns end nodes from graph G with degree == 1 """
    return [n for n in nx.nodes(g) if nx.degree(g, n) == 1]
    # return [n for n in nx.nodes_iter(g) if nx.degree(g, n) == 1]

def connected_component_to_graph(img,wholeGraph):
    """Input: Centerline
    Takes each connected component from centerline, buildTree creates graph, mergeEdges merges
    neighbouring pixels and forms longer edges joining between pixels, removeSmallEdge removes
    false edges, joins all individual graphs using union function. Finally removes isolated nodes"""
    label_im, nb_labels = scipy.ndimage.label(img, structure=cube(3))
    unique, counts, indices = np.unique(label_im, return_counts=True, return_index=True)
    # print (unique, counts, indices)


    for each_label in range(0,nb_labels+1):
        if counts[each_label] != 0:
            # print ("Label ID of component",each_label)
            each_component = np.where(label_im == unique[each_label], label_im, 0)
            index = np.where(label_im == each_label)
            startAt = [x[0] for x in index]
            print ("Starting point of Graph",startAt)
            componentGraph, G = buildTree(each_component, start=startAt)
            mergedGraph = mergeEdges(G)
            # mergedGraph = G #graph with Allnodes between start and end nodes for each branch

            trimmedGraph = removeSmallEdge(mergedGraph)
            remergedGraph = mergeEdges(trimmedGraph) #trimmedGraph

            wholeGraph = nx.union(wholeGraph , G)  #remergedGraph
    wholeGraph.remove_nodes_from(list(nx.isolates(wholeGraph))) #remove isolated nodes without neighbouring nodes
    # network_plot_3D(wholeGraph, 0, img, 0, linethickness=1, save=False, csa=False)

    return wholeGraph


def network_plot_3D(G, angle,img,fullVessel,linethickness=1, save=False,csa = False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()
    print ("number"+str(n))
    # Get the maximum number of edges adjacent to a single node
    # edge_max = max([G.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    # colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi,  s=20, alpha=0.7,c='blue') #c=colors[key]


        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            if csa == False:
                # Plot the connecting lines
                ax.plot(x, y, z, c='black', alpha=0.5, linewidth=linethickness)

    # Set the initial view
    ax.view_init(30, angle)

    # x1,y1,z1 = img.nonzero()
    # ax.scatter(x1, y1, z1, zdir='z', c='red',alpha=0.2,s=4)

    # x2,y2,z2 = fullVessel.nonzero()
    # ax.scatter(x2,y2,z2, zdir='z', c='green',alpha=0.1,s=2)
    # Hide the axes
    # ax.set_axis_off()
    # if save is not False:
    #     # plt.savefig("C:\scratch\\data\" +str(angle).zfill(3)+".png")
    #     plt.close('all')
    #     else:
    #     plt.show()
    plt.show()


    return

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
        distance = distance + np.linalg.norm(np.array(p) - np.array(q))
        print(np.linalg.norm(np.array(p) - np.array(q)))
    print(a)
    print(pairs)
    print("distance",distance,"weight",len(a))
    return distance

def pointPicking (pixelList,weight):
    """returns list of pixels between 10-90% of whole list"""
    newList = pixelList[int(len(pixelList) * .0): int(len(pixelList) * .99)]
    return newList

def radialDistance(middlePixels,new_img_dst): #new_img_dst
    """Uses middle pixels and distance transform of vessel mask to return distance of centerline
    pixel from the nearest background, thereby calculating the radial distance of each vessel segment.
    Returns cross-sectional area and average radius from top 20% of vessel radius"""
    radii = []
    for i in range(0,len(middlePixels)):
        R = middlePixels[i]
        # radialDist = new_img_dst[R[0],R[1],R[2]]  #new_img_dst
        radialDist = new_img_dst[str(R)]  #new_img_dst

        print("radius",radialDist)
        radialDist = int(radialDist)
        radii.append(radialDist)
    if len(radii) != 0:
        radii.sort(reverse=True)
        # toptrim = stats.trim1(radii,proportiontocut=0.1,tail='right')
        trimmedradius = stats.trim1(radii,proportiontocut=0.9,tail='left')
        avg_radius = statistics.mean(trimmedradius)
    else:
        avg_radius = 0
    crossSecArea = pi * avg_radius ** 2
    return crossSecArea,avg_radius

def avgBranchRadius (combinedGraph, new_img_dst,img):#,new_img_dst,img):  #new_img_dst
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
        allpoints.append(u.point)
        allpoints.append(o.pixels)
        allpoints.append(v.point)
        pixelDist = pixelDistance(allpoints)

        if weight >= 5:
            betweenPxlList = pointPicking(o.pixels,weight)
            csaAvg, radiusAvg = radialDistance(betweenPxlList,new_img_dst)  #new_img_dst
            ##################################################################

            combinedGraph[u][v]["length"] = pixelDist #+ endNodeAddition In my small vascular tree, I do not need endnodeadddition since the center line in my example start from center of first circle
            combinedGraph[u][v]["radius"] = radiusAvg
            combinedGraph[u][v]["csa"] = csaAvg
            # combinedGraph[u][v]["volume"] = volume


            ###################################################################

        elif weight < 5 and weight >= 2:
            mid = weight//2
            csaSmall,radiusSmall = radialDistance(o.pixels[mid-1:mid+1],new_img_dst)  #new_img_dst
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

def resample_boston(image, new_spacing=[1, 1, 1]):
    """
    Determine current pixel spacing
    Resample the scan with 1,1,1 spacing to address pixels resolution variation
    """

    # head = scan.get_header()
    spacing = np.array([0.5, 0.6641, 0.6641])
    print (spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print ("spacing and new spacing,real resize",spacing,new_spacing,real_resize_factor)

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

def writeGraph(G):
    """Write edges of graph in xml format"""


    nx.write_edgelist(G, "small_tree_CIP_coords_data_3branches.csv",  delimiter=' ') # data=['length', 'radius']




################################ run the code ###########################################################

# dir = '/eresearch/lung/arah607/COPDgene/CIPtest/Normals/19003F/COPD1/19003F_INSP_B31f_340_COPD1'
#
# file = 'CTparticles.vtk_rightLungVesselParticles.vtk'
#
# mesh = pv.read(dir + '/' + file)
# #
#
# img = mesh.points[:]
# img = np.array(img).astype(np.uint8)
# # vessel_extraction.visualizeNifti(img)
headerValue = [0, 1, 2]
img = pd.read_csv("small_tree_CIP_coords_55branches.csv",header=None) # read the csv file radius 20mm L=20
# img = pd.read_csv("test1.csv",header=None) # read the csv file radius 20mm L=20


# img = pd.read_csv("test_tree_rounded.csv",header=headerValue) # read the csv file radius 20mm L=20


# img = pd.read_csv("tessssttttt.csv",header=headerValue) # read the csv file radius 20mm L=20

fullVessel = np.zeros((350,350,600),dtype=np.uint32)
Cordinate_inf = {}
radii_coords = {}
xaxis=0
yaxis=0
zaxis=0
for index, row in img.iterrows(): # find the coordinate in csv file and read it (like: 25, 120, 300))
    fullVessel[row[0], row[1], row[2]] = 1
    if row[0] <0:
        xaxis = fullVessel.shape[0] + row[0]
    else:
        xaxis = row[0]
    if row[1] <0:
        yaxis = fullVessel.shape[1] + row[1]
    else:
        yaxis = row[1]

    if row[2]<0:
        zaxis = fullVessel.shape[2] + row[2]
    else:
        zaxis = row[2]




    val_indx = '[' + str(row[0]) + ', ' + str(row[1]) + ', '+  str(row[2]) + ']'
    transformed_indx = '[' + str(xaxis) + ', ' + str(yaxis) + ', ' + str(zaxis) + ']'
    Cordinate_inf[val_indx]= transformed_indx
    # radii_coords[transformed_indx] = str(row[3])

# vessel_extension_bool = np.array(scipy.ndimage.binary_closing(fullVessel,iterations=1,structure=ball(5)).astype(np.int32)) #remove the holes in branches
fullVessel = np.where(fullVessel, 1, 0).astype(np.int32)# the value for vessels section area is one value and the background is zero value
# img = centerline_extraction(fullVessel) #the skimage library calculate the centerline of the each branch vessel
lung_mask_im = np.array(fullVessel).astype(np.uint8)
img = lung_mask_im
# img_resampled = (resample_boston(lung_mask_im)).astype(np.uint8)
# img = img_resampled
# vessel_extraction.visualizeNifti(img)




H = nx.Graph()
# imgSkT = img.copy()

# # lung_mask_im = np.array([np.array(Image.open(fname)) for fname in img])  # all boston vessels(each slice), the whole is 3 dimension array
# # lung_mask_im = np.where(lung_mask_im == 1, 1, 0)
# fullVessel = np.zeros((600,600,500),dtype=np.uint32)
# for i in range(0, len(img)):
#     fullVessel[img[i][0], img[i][1], img[i][2]] = 1
# vessel_extension_bool = np.array(scipy.ndimage.binary_closing(fullVessel,iterations=1,structure=ball(5)).astype(np.int32)) #remove the holes in branches
# fullVessel = np.where(vessel_extension_bool, 1, 0).astype(np.int32)# the value for vessels section area is one value and the background is zero value
# # img = centerline_extraction(fullVessel) #the skimage library calculate the centerline of the each branch vessel
# lung_im = np.array(fullVessel).astype(np.uint8)
# img = lung_im

#convertion of centerline to graph-based data structure
combined_graph = connected_component_to_graph(img,H)

# vessel_extraction.visualizeNifti(combined_graph)
network_plot_3D(combined_graph,img=img,fullVessel=fullVessel, angle=0)

# vertices, faces, normals , values = skimage.measure.marching_cubes_lewiner(fullVessel)
# mesh3d = vv.mesh(vertices, faces, normals, values)
# vv.use().Run()
# mesh.point_data.active_scalars_name = 'scale'
#
# mesh.plot()
new_img_dst1 = np.zeros((350,350,800),dtype=np.uint32)
new_img_dst = radii_coords


newCombinedGraph = avgBranchRadius(combined_graph, new_img_dst, img)  #normal_radii

writeGraph(newCombinedGraph)


