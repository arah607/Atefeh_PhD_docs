import math

import time

import statistics

import numpy as np

from numpy.linalg import norm

import os

import pymesh

import pyvista as pv

import glob
import scipy
import nibabel as nib
import numpy as np
from skimage.morphology import cube
# import vessel_extraction
# import VesselVolumeCalc
from PIL import Image
import scipy.ndimage
import skimage.filters
from skimage.filters import window
import networkx as nx
import matplotlib.pyplot as plt
# import natsort
import csv
import pandas as pd

##########################################################

def export_data(name, vertices):
    opf = open(name + '.exdata', 'w+')

    opf.write(" Group name: from_cip\n")

    opf.write(" #Fields=1\n")

    opf.write(" 1) coordinates, coordinate, rectangular cartesian, #Components=3\n")

    opf.write("   x.  Value index= 1, #Derivatives=0\n")

    opf.write("   y.  Value index= 2, #Derivatives=0\n")

    opf.write("   z.  Value index= 3, #Derivatives=0\n")

    for nv in range(len(vertices)):
        opf.write(" Node: %12d\n" % (nv + 1))

        opf.write("   %10.3f  %10.3f  %10.3f\n" % (vertices[nv][0], vertices[nv][1], vertices[nv][2]))

    opf.close()


###################################################################


def export_radii(mesh_radii, data):
    opf = open(mesh_radii + '.exdata', 'w+')

    opf.write(" Group name: from_cip\n")

    opf.write(" #Fields=1\n")

    opf.write(" 1) radius, field, rectangular cartesian, #Components=1\n")

    opf.write("   radius.  Value index=1, #Derivatives=0\n")

    for nv in range(len(data)):
        opf.write(" Node: %12d\n" % (nv + 1))

        opf.write("   %10.3f\n" % (data[nv]))

    opf.close()


def export_RegionType(mesh_RegionType, data):
    opf = open(mesh_RegionType + '.exdata', 'w+')

    opf.write(" Group name: from_cip\n")

    opf.write(" #Fields=1\n")

    opf.write(" 1) RegionType, field, rectangular cartesian, #Components=1\n")

    opf.write("   RegionType.  Value index=1, #Derivatives=0\n")

    for nv in range(len(data)):
        opf.write(" Node: %12d\n" % (nv + 1))

        opf.write("   %10.3f\n" % (data[nv]))

    opf.close()
#################################################################



# dir = '/eresearch/lung/arah607/COPDgene/CIPtest/Normals/19003F/COPD1/19003F_INSP_B31f_340_COPD1'
# dir = '/eresearch/lung/arah607/COPDgene/CIPtest/Normals/19020F/COPD1/19020F_INSP_B31f_300_COPD1'
dir = '/eresearch/lung/arah607/COPDgene/COPDgene_Phase1_minus2_Normals/AV'

# dir = '/eresearch/lung/arah607/COPDgene/COPDgene_Phase1_minus2_Normals/CIP_completed/Outputs'



# file = '15814W_vesselPartial.vtk'
# file = 'CTparticles.vtk_rightLungVesselParticles.vtk'
# file = '15814W_wholeLungVesselParticles.vtk'
# file = '16032X_wholeLungVesselParticles.vtk'
# file = '16311B_wholeLungVesselParticles.vtk'
# file = '16315J_wholeLungVesselParticles.vtk'
# file = '16617Z_wholeLungVesselParticles.vtk'
# file = '17257A_wholeLungVesselParticles.vtk'
# file = '17275C_wholeLungVesselParticles.vtk'
# file = '17929X_wholeLungVesselParticles.vtk'
# file = '18347G_wholeLungVesselParticles.vtk'
# file = '18615F_wholeLungVesselParticles.vtk'
file = '19020F_wholeLungVesselParticles.vtk'

mesh = pv.read(dir + '/' + file)




mesh.point_data.active_scalars_name = 'scale'
# mesh.point_data.active_scalars_name = 'ChestRegionChestType'





# vertices = mesh.point_data['hevec1']
radii = mesh.point_data['scale']
RegionType = mesh.point_data['ChestRegionChestType']

# Artery = np.where(RegionType == 12800, 1, 0)

# Find vein values and change them from 13056 to 1 and other values are zero
# vein = np.where(RegionType == 13056, 2, 0)

print(mesh)

print(mesh.point_data)

# radii = mesh.point_data.active_scalars

# export_data('mesh_coords_19020F', mesh.points[:])
#
# export_radii('mesh_radii_19020F', radii[:])

export_data('/home/arah607/Desktop/19020F/19020F_wholeLungVesselParticles_coords', mesh.points[:])
export_radii('/home/arah607/Desktop/19020F/19020F_wholeLungVesselParticles_radii', radii[:])
export_RegionType('/home/arah607/Desktop/19020F/19020F_wholeLungVesselParticles_RegionType', RegionType[:])

mesh.plot()











