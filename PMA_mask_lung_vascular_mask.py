import glob
import os

#import itk
import itk.itkBinaryThinningImageFilterPython

import DenoisingVessels
from csv import writer

import pandas as pd
import networkx as nx
import nibabel as nib
import natsort
import numpy as np
from PIL import Image
from skimage import morphology
import scipy.ndimage.interpolation
import vtk_nifit_render
import scipy
from skimage.morphology import cube
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def resample(image, new_spacing=[1, 1, 1, 1]):
    """
    Determine current pixel spacing
    Resample the scan with 1,1,1 spacing to address pixels resolution variation
    """

    # head = scan.get_header()
    spacing = np.array([0.5, 0.625, 0.625, 1])
    print (spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print ("spacing and new spacing,real resize",spacing,new_spacing,real_resize_factor)

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image


# def centerline_extraction (mask):
#     """ Input takes 3D ndarray of image mask
#         Returns 3D centerline after skeletonization
#     """
#     from skimage import morphology
#     out_skeletonize = morphology.skeletonize_3d(mask)
#     return out_skeletonize

def centerline_extraction (mask):
    """ Input takes 3D ndarray of image mask
        Returns 3D centerline after skeletonization
    """
    import sgext
    sgext_image = sgext.itk.IUC3P()
    sgext_image.from_pyarray(lung_mask_img)
    thin_image = sgext.scripts.thin(input=sgext_image, tables_folder=sgext.tables_folder, skel_type="end",
                                    select_type="first", persistence=2, visualize=False, verbose=True)

    # thin_filename = "/hpc/arah607/CETPH1.nrrd"
    # thin_filename = "/hpc/arah607/testnormalMPA.nrrd"
    thin_filename = "/hpc/arah607/Alfred8_Post_data.nrrd"


    image_img_trunk = sgext.itk.write(thin_image, thin_filename)

    return image_img_trunk


def Read_centerline(image):
    import nrrd
    filename = '/hpc/arah607/CETPH6.nrrd'
    readdata, header = nrrd.read(filename)
    trunck_mask_array = np.where(readdata != 0, 1, 0)
    trunck_mask_array = np.where(trunck_mask_array == 1, 1, 0)

    return  trunck_mask_array



class Vertex:
    """ Class to define vertices of the graph"""
    def __init__(self, point, degree=0, edges=None):
        # self.point = np.asarray(point)
        self.point = point # show node
        self.degree = degree # show the number of the edge that is coming from one node
        self.edges = [] # show edge
        self.visited = False# it hasnt read any node yet. it has false value as default.
        if edges is not None:
            self.edges = edges

    def __str__(self):
        return str(self.point)

#Class to define edges of the graph
class Edge:
    """Class to define edges of the graph"""
    def __init__(self, start, end=None, pixels=None):
        self.start = start # starting point with [x y z]
        self.end = end # ending point with [x y z]
        self.pixels = [] # each point with [xn yn zn] is between start and end points.
        if pixels is not None:
            self.pixels = pixels
        self.visited = False
#subroutine to get 3x3x3 connected neighbour offsets
def get_neighbours(p, exclude_p=False, shape=None):
    """subroutine to get 3x3x3 connected neighbour offsets
    (this function will give the list of all neighbours from that center point (vertex), give us node location.
    for example if we consider the one center node in the cube, we can underestand how many neighbours doest exist for that
      center node (neighbours are like cube for each center point). After that, this function just consider one neighbour with one value for that center node, because
      the other neighbour nodes are zero and there is only one connecting node (we consider the centerline and there is only one center point(pixel) in each direction))"""

    ndim = len(p) # P this the center node here

    # generate an (m, ndims) array containing all strings over the alphabet {0, 1, 2}:
    offset_idx = np.indices((3,) * ndim).reshape(ndim, -1).T

    # use these to index into np.array([-1, 0, 1]) to get offsets
    offsets = np.r_[-1, 0, 1].take(offset_idx)

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
    class Vertex and added to graph (the list of the nodes), similarly edges are created and added.....
    get all pixel with value one, the background is zero (black color). This function convert pixel with value one to node, and then makes it black and goes to the next. because when it do it
     iteratively the node as a black is not come again to that. Get all tree and makes each pixel to v. Now we have list of the V and list  of the E  """
    # copy image since we set visited pixels to black
    img = img.copy() # copy of the centerline
    img[np.nonzero(img)] = 1 #the lable of the centerline is one(change all values to one)

    shape = img.shape
    nWhitePixels = np.sum(img) #total number of one value

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

    G = nx.Graph()

    blackedPixels = 0 # now the number of the blackpixel is zero. But the white pixels convert to black, the number of the blackpixels should for example 2000
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
        img[startV.point[0], startV.point[1], startV.point[2]] = False # convert white to black. point[0]=x, point[1]=y, point[3]=z
        blackedPixels += 1

        # create a new graph
        # G = nx.Graph()

        G.add_node(startV,pos=startV.point) # add start node to graph with its position
        # G.add_node(startV, x=startV.point[0], y=startV.point[1], z=startV.point[2])

        # build graph in a breath-first manner by adding
        # new nodes to the right and popping handled nodes to the left in queue
        while len(queue): # at least there is startV in queue
            currV = queue[0]  # current vertex is queue[0], and it means we can have access to first value of the queue
            # print("Current vertex: ", currV.point)

            # start with V and, check all neighbour pixels with value one
            for nbOff in nbPxOff:

                # pixel index+-

                # print currV.point
                pxIdx = currV.point + nbOff # Check each neighboures with moving from current vertext based on nboff

                # print currV.point,pxIdx

                if (pxIdx[0] < 0 or pxIdx[0] >= shape[0]) or (pxIdx[1] < 0 or pxIdx[1] >= shape[1]) \
                        or (pxIdx[2] < 0 or pxIdx[2] >= shape[2]):
                    continue  # some neighbour pixels around v are out of the image bond and we do not consider them. Continue stops while loop

                if img[pxIdx[0], pxIdx[1], pxIdx[2]]: # if neighbour is inside, then we make it as a new vertex with x y z
                    # print( "nb: ", pxIdx, " white ")
                    # pixel is white
                    newV = Vertex([pxIdx[0], pxIdx[1], pxIdx[2]])


                    # add edge from currV <-> newV
                    G.add_edge(currV, newV, object=Edge(currV, newV))
                    # G.add_edge(newV,currV)

                    # add node newV
                    G.add_node(newV, pos=newV.point)
                    # G.add_node(newV, x = startV.point[0], y = startV.point[1], z = startV.point[2])


                    # push vertex to queue. now queue has two members
                    queue.append(newV)

                    # set neighbor pixel to black
                    img[pxIdx[0], pxIdx[1], pxIdx[2]] = False
                    blackedPixels += 1

            # pop currV (means remove current vertex). for each neighbour in the list of neighbours do this, then popleft makes that neighbour pixel to zero and goes for another neighbour
            #Maybe I should change this connection, THinkkkkkkkkkkkk
            #for example queue has three elements v1 v2  v3, after pop v1 then the lenght of queue is not zero because it has two elements. again same happen for v2(makes it vertex, added to graph,
            # create edge and pop it off. It continue until neighbours will finish
            queue.popleft()
        # end while

        # empty queue
        # current graph is finished ->store it
        graphs.append(G)
        G_composed = nx.compose_all(graphs)

        # network_plot_3D(G, 0, img, 0, linethickness=1, save=False,csa = False)

        # display(G)


        # reset start
        start = None
        # H = nx.compose(H, G)
    # network_plot_3D(G, 0, img, 0, linethickness=1, save=False, csa=False)
    # end while
    # mlab.show()

    return G

def mergeEdges(graph):

    # copy the graph
    g = graph.copy()

    # start not at degree 2 nodes, and it shows it is a middle node. start node has degree one, or degree three (in biforcation). Middle node has degree two
    startNodes = [startN for startN in g.nodes() if nx.degree(g, startN) != 2]


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

                # fuse the pixel list from right and left and add our pixel n.point
                g.add_edge(v0, nbs[1],
                           object=Edge(v0, nbs[1], pixels=pxL0 + [v1.point] + pxL1))

                # delete the node n
                g.remove_node(v1)

                # set v1 to new left node
                v1 = nbs[1]

            else:
                counter += 1
                if counter == len(startNNbs):
                    break
                v1 = startNNbs[counter]  # next nb of v0

    # weight(lenght) the edges according to their number of pixels
    for u, v, o in g.edges(data="object"):
        g[u][v]["weight"] = len(o.pixels)



    # network_plot_3D(g, 0)
    # display(g)
    return g

def getEndNodes(g):
    """Returns end nodes from graph G with degree == 1 """
    return [n for n in nx.nodes(g) if nx.degree(g, n) == 1]

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

    for v0 in startNodes:
        # print("v0",v0)
        # start a line traversal from each neighbor
        startNNbs = list(nx.neighbors(g, v0))
        # print("neighbors",startNNbs)
        #for a node both neighbor and ending
        chosen = [x for x in g.nodes() if x in endNodes and x in startNNbs]

        for x0 in chosen:
            if g[v0][x0]["weight"] < 45:
                # print("chosen weight",g[v0][x0]["weight"])
                g.remove_node(x0)


    return g

def visualizeNifti (array):
    """3D image array is converted to nifti format using custom made target affine. This is visualised
    using VTK volume rendering"""
    array = np.array(array).astype(np.uint8)
    targetaffine4x4 = np.eye(4)
    targetaffine4x4[3,3] = 1
    nifti_image_from_array = nib.Nifti1Image(array, targetaffine4x4)
    save_path = '/home/arah607/Joyce/Vessel_Quantification/tempfile.nii'
    nib.save(nifti_image_from_array, save_path)
    vtk_nifit_render.vtk_pipeline(save_path)
    # vtk_nifit_render.vtk_pipeline(vesselmaskpath)

    # print (nifti_image_from_array.affine)



# def network_plot_3D(G, angle,linethickness=1, save=False,csa = False):
#     # Get node positions
#     pos = nx.get_node_attributes(G, 'pos')
#
#     # Get number of nodes
#     n = G.number_of_nodes()
#     print ("number"+str(n))
#     # Get the maximum number of edges adjacent to a single node
#     #edge_max = max([G.degree(i) for i in range(0,n)])
#     # Define color range proportional to number of edges adjacent to a single node
#     #colors = [plt.cm.plasma(G.degree(i) / edge_max) for i in range(n)]
#     # 3D network plot
#     with plt.style.context(('ggplot')):
#
#         fig = plt.figure(figsize=(8, 8))
#         ax = Axes3D(fig)
#
#         # Loop on the pos dictionary to extract the x,y,z coordinates of each node
#         for key, value in pos.items():
#             xi = value[0]
#             yi = value[1]
#             zi = value[2]
#
#             # Scatter plot
#             ax.scatter(xi, yi, zi, s=20, alpha=0.7,c='blue') #c=colors[key]
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
#     #ax.view_init(30, angle)
#
#     # x1,y1,z1 = img.nonzero()
#     # ax.scatter(x1, y1, z1, zdir='z', c='red',alpha=0.2,s=4)
#     for angle in range(0, 360):
#         ax.view_init(30, angle)
#         plt.draw()
#         plt.pause(.001)
#     # x2,y2,z2 = fullVessel.nonzero()
#     # ax.scatter(x2,y2,z2, zdir='z', c='green',alpha=0.1,s=2)
#     # Hide the axes
#     # ax.set_axis_off()
#     # if save is not False:
#     #     # plt.savefig("C:\scratch\\data\" +str(angle).zfill(3)+".png")
#     #     plt.close('all')
#     #     else:
#     #     plt.show()
#     plt.show()
#     # plt.close()
#
#     return

def network_plot_3D(G):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in G])
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


    nx.write_edgelist(G, "trunk_coors.csv",  delimiter=' ')

################################################ RUN #######################################################
# direct_mask_path = glob.glob("/hpc/arah607/CTEPH1/*.jpg") # manual segmentation and save as a jpg format
# # direct_mask_path = glob.glob("/hpc/arah607/CTEPH6/*.jpg")
#
# # direct_mask_path = glob.glob("/hpc/arah607/CTEPH1/DICOMS.labels.nrrd")
# direct_mask_path = natsort.natsorted(direct_mask_path)
# # Go through to each slice and store each slice as an array
# lung_mask = np.array([np.array(Image.open(fname)) for fname in direct_mask_path])
# lung_mask_array = np.where(lung_mask != 0, 1, 0)
# lung_array = np.zeros((231, 512, 512))
# for i in range(lung_mask_array.shape[0]):
#     for j in range(lung_mask_array.shape[1]):
#         for k in range(lung_mask_array.shape[2]):
#             if lung_mask_array[i][j][k][0] == 0 and lung_mask_array[i][j][k][1] == 0 and lung_mask_array[i][j][k][2] == 0:
#                 lung_array[i, j, k] = 0
#             else:
#                 lung_array[i, j, k] = 1


import nrrd
# filename1 = '/hpc/arah607/pul_tru.nrrd'
# filename1 = '/hpc/arah607/test_trunk_segment.nrrd'
# filename1 = '/hpc/arah607/MPA-001-post.nrrd'
# filename1 = '/hpc/arah607/testnormalMPA.nrrd'
# filename1 = '/hpc/arah607/test_subone_tru.nrrd'
filename1 = '/hpc/arah607/Alfred8_Post.nrrd'

lung_array, header1 = nrrd.read(filename1)
lung_mask_array = np.where(lung_array == 1, 1, 0)
lung_mask = scipy.ndimage.binary_closing(lung_mask_array, structure=cube(5)) # Find the gaps (the zero values between ones) and change their values from zero to one to fill the gaps
lung_mask_img = np.where(lung_mask, 1, 0).astype(np.uint8)
# lung_mask_array = lung_mask_array.shape[:-1]
# array_resampled = (resample(lung_mask_array)).astype(np.uint8)
# trunk = array_resampled
# visualizeNifti(lung_mask_img)
center_trunk = centerline_extraction(lung_mask_img)# the centerline is saved in img
import nrrd
filename = '/hpc/arah607/CETPH1.nrrd'
# filename = '/hpc/arah607/testnormalMPA.nrrd'
# filename = '/hpc/arah607/Alfred8_Post_data.nrrd'
readdata, header = nrrd.read(filename)
trunck_mask_array = np.where(readdata != 0, 1, 0)
trunck_mask_array = np.where(trunck_mask_array == 1, 1, 0)

visualizeNifti(trunck_mask_array)

G = buildTree(trunck_mask_array, start=None)
network_plot_3D(G)
writeGraph(G)

mergedGraph = mergeEdges(G)
network_plot_3D(mergedGraph)
trimmedGraph = removeSmallEdge(mergedGraph)
network_plot_3D(trimmedGraph)
mergedGraph_new = mergeEdges(trimmedGraph)
network_plot_3D(mergedGraph_new)


writeGraph(mergedGraph)
# H = nx.Graph()
# startAt = [x[0] for x in index]  # x[0] can start from wherever
# started_component_list.append(np.asarray(startAt))
# print("Starting point of Graph", startAt)
# G = buildTree(center_trunk, start=None)  # in buildTree we have just one connect tree(small tree)
# mergedGraph = mergeEdges(G)  # graph with nodes and branches
# # trimmedGraph = removeSmallEdge(mergedGraph) # I should remove this line

# itk.itkBinaryThinningImageFilterPython.binary_thinning_image_filter()
#itk.binary_thinning_image_filter(image)



# import sgext
# sgext_image = sgext.itk.IUC3P()
# sgext_image.from_pyarray(lung_mask_img)
# thin_image = sgext.scripts.thin(input=sgext_image,
#                    tables_folder= sgext.tables_folder,
#                    skel_type="end",
#                    select_type="first",
#                    persistence=2,
#                    visualize=False,
#                    verbose=True
#                    )
#
# thin_filename ="/hpc/arah607/thin_image.nrrd"
# sgext.itk.write(thin_image, thin_filename)
#
# import nrrd
# filename = '/hpc/arah607/thin_image.nrrd'
# readdata, header = nrrd.read(filename)
# lung_mask_array = np.where(readdata != 0, 1, 0)
# lung_mask_array = np.where(lung_mask_array == 1, 1, 0)
# visualizeNifti(lung_mask_array)
