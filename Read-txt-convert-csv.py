from csv import writer
# path = '/home/arah607/Desktop/small_tree_CIP_coords_5branches.exdata'
# path = '/home/arah607/Desktop/small_tree_CIP_coords_6branches.exdata'
# path = '/home/arah607/Desktop/small_tree_CIP_coords_5branches_radii.exdata'
# path = '/home/arah607/Desktop/test_small_lung.exdata'
# path = '/home/arah607/Desktop/whole_right_lung.exdata'

# path = '/home/arah607/Desktop/mpa_cip.exnode'
# path = '/home/arah607/Desktop/right_lung_artery.exnode'
# path = '/home/arah607/Desktop/right_lung_artery_neww.exnode'
# path = '/home/arah607/Desktop/right_lung_artery_onebranch.exnode'
# path = '/home/arah607/Desktop/right_mainartery_CIP.exnode'
# path = '/home/arah607/Desktop/right_onesmall_branch_new.exnode'
# path = '/home/arah607/Desktop/test_branch_abovelope.exnode'
# path = '/home/arah607/Desktop/right_lung_artery_newwww.exnode'
# path = '/home/arah607/Desktop/only_trunk_coors.exnode'
# path = '/home/arah607/Desktop/small_upper_branch.exnode'
# path = '/home/arah607/Desktop/top_branch_9Feb.exnode'
# path = '/home/arah607/Desktop/all_node_top_test.exnode'
# path = '/home/arah607/Desktop/top_filter_14feb.exnode'
# path = '/home/arah607/Desktop/normal_subone_alldata.exdata'
#path = '/home/arah607/Desktop/3d_digit_creatingpoints.exnode'
# path = '/hpc/arah607/3d_digitise_creatingpoints/allnodes_radius_19003F.exdata'
# path = '/home/arah607/Desktop/modify_trunk_mainpul_points.exnode'
# path = '/home/arah607/Desktop/onebranch_trunkmain_points.exnode'
# path = '/home/arah607/Desktop/allnodes_filter_radiibigger1.5.exnode'
# path = '/home/arah607/Desktop/modify_trunk_mainpul_points_27April.exnode'
# path = '/home/arah607/Desktop/trunk_CIPdata_1may.exnode'
# path = '/home/arah607/Desktop/test_1May.exnode'
# path = '/home/arah607/Desktop/19020F/trunknodes_mainartery_19020F.exdata'



#path = '/home/arah607/Desktop/allnodes_19020F.exdata'


############################COPDGENnormal#########################
# path = '/home/arah607/Desktop/15814w_lung_coor_ArteryVein.exdata'
# path = '/home/arah607/Desktop/15814w_lung_coor_radii.exdata'

# path = '/home/arah607/Desktop/test_twobranches_15814w.exnode'
# path = '/home/arah607/Desktop/16032X/allnodes_radii_ArteryVein16032X.exnode'
# path = '/home/arah607/Desktop/16032X/16032X_wholeLungVesselParticles_radii_RegionType.exnode'
# path = '/home/arah607/Desktop/16311B/16311B_wholeLungVesselParticles.exnode'
# path = '/home/arah607/Desktop/16315J/16315J_wholeLungVesselParticles.exnode'
# path = '/home/arah607/Desktop/16617Z/16617Z_wholeLungVesselParticles.exnode'
# path = '/home/arah607/Desktop/test_15814W.exdata'
# path = '/home/arah607/Desktop/smalltest_15814W.exnode'
# path = '/home/arah607/Desktop/left_test_15814W.exnode'
# path = '/home/arah607/Desktop/example_15814w.exnode'
# path = '/home/arah607/Desktop/test_small_15814w.exnode'
# path = '/home/arah607/Desktop/test_15814w_allnodes.exnode'
path = '/home/arah607/Desktop/18615F.exdata'

# path = '/home/arah607/Desktop/test_coooords.exdata'
# file = open(path)
# read_line= [7]
# for pos, line in enumerate(file):
#     if pos in read_line and type(line) == str:
#         read_line = read_line[0]+1
#         read_line = [read_line]
#         pos = read_line
#         for pos, line in enumerate(file):
#             line = list(line)
#             line[2] = int(line[2])
#             if line[2] == int(line[2]):
#
#                 a = line
#                 print(a)
#

file_text = open(path, "r")

# for i in range(0,7):
for i in range(0,9):
# for i in range(0,7):
    file_line = file_text.readline()
# with open('small_tree_CIP_coords_5branches_radii.csv', 'w', newline='') as data_file:
# with open('left_lung.csv', 'w', newline='') as data_file:
# with open('right_mainartery_CIP.csv', 'w', newline='') as data_file:
# with open('whole_right_lung_radiusfilter.csv', 'w', newline='') as data_file:
# with open('normal_subone_alldata.csv', 'w', newline='') as data_file:
# with open('whole_rightlung_3F.csv', 'w', newline='') as data_file:
# with open('onebranch_trunkmain_points.csv', 'w', newline='') as data_file:
with open('/home/arah607/Desktop/18615F.csv', 'w', newline='') as data_file:
    # with open('tessssttttt.csv', 'w', newline='') as data_file:

    csv_writer = writer(data_file)
    while True:
        file_line = file_text.readline()
        if file_line.find("Node") != -1:
            Write_number=[]
            for i in range(0,5):
                file_line = file_text.readline()
                Write_number.append(float(file_line))
            # csv_writer.writerow(Write_number)
            if Write_number[4] >1.5 and Write_number[0] == 12800: #1.5
                csv_writer.writerow(Write_number)
        if not file_line:
            print("End Of File")
            break

file_text.close()







############################33 conver dicom file to jpg file ###############################################

# import os
# import pydicom
# import numpy as np
# from PIL import Image
#
# # Define input and output directories
# input_dir = '/hpc/arah607/lung/Data/Normal/19020F/FRC/dicom/'
# output_dir = '/hpc/arah607/lung/Data/Normal/19020F/FRC/Raw'
# count=0
# # Loop through all DICOM files in input directory
# for filename in os.listdir(input_dir):
#     if filename.endswith('.dcm'):
#         # Read DICOM file
#         ds = pydicom.dcmread(os.path.join(input_dir, filename))
#         count = count + 1
#         # Convert pixel data to 8-bit format
#         pixel_data = ds.pixel_array.astype(np.uint8)
#
#         # Convert pixel data to image
#         img = Image.fromarray(pixel_data)
#
#         # Define output filename
#         output_filename = os.path.splitext(filename)[0] + '.jpg'
#
#         # Save image to output directory
#         img.save(os.path.join(output_dir, output_filename))




# import os
# import pydicom
# import numpy as np
# import cv2# Function to convert pixel data to Hounsfield Units (HU)
# def pixel_to_hu(image, dicom):
#     slope = dicom.RescaleSlope
#     intercept = dicom.RescaleIntercept
#     hu_image = image * slope + intercept
#     return hu_image# Function to flip image vertically
# def flip_vertical(image):
#     flipped = cv2.flip(image, 0)
#     return flipped# Define input and output paths
# input_path = '/hpc/arah607/lung/Data/Human_Lung_Atlas/19020F/FRC/dicom/'
# output_path = '/hpc/arah607/lung/Data/Human_Lung_Atlas/19020F/FRC/Raw'
#
# dicom_files = sorted(dicom_files, key=lambda x: int(os.path.splitext(x)[0]))
# for i, file in enumerate(os.listdir(input_path)):
#     if file.endswith(".dcm"):
#         # Load DICOM file and extract pixel data
#         dicom = pydicom.dcmread(os.path.join(input_path, file))
#         pixel_data = dicom.pixel_array        # Convert pixel data to Hounsfield Units (HU)
#         hu_image = pixel_to_hu(pixel_data, dicom)        # Flip image vertically
#         flipped_image = flip_vertical(hu_image)        # Scale pixel values to 0-255 and convert to 8-bit unsigned integer
#         scaled_image = np.interp(flipped_image, (flipped_image.min(), flipped_image.max()), (0, 255)).astype(np.uint8)        # Save image as high-quality JPG file
#         output_file = 'raw%.4d.jpg' % i #+ os.path.splitext(file)[0]
#         cv2.imwrite(os.path.join(output_path, output_file), scaled_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#



############################################ conver dicom file to jpg file ############################
import os
import pydicom
import numpy as np
import cv2# Function to convert pixel data to Hounsfield Units (HU)
def pixel_to_hu(image, dicom):
    slope = dicom.RescaleSlope
    intercept = dicom.RescaleIntercept
    hu_image = image * slope + intercept
    return hu_image# Function to flip image vertically
def flip_vertical(image):
    flipped = cv2.flip(image, 0)
    return flipped# Define input and output paths
input_path = '/hpc/arah607/lung/Data/Human_Lung_Atlas/19020F/FRC/dicom/'
output_path = '/hpc/arah607/lung/Data/Human_Lung_Atlas/19020F/FRC/Raw'
dicom_files = [file for file in os.listdir(input_path) if file.endswith(".dcm")]# Sort the DICOM files in numerical order
dicom_files = sorted(dicom_files, key=lambda x: int(os.path.split(x)[0]))# Loop through all sorted DICOM files
i = 0
for file in dicom_files:
    # Load DICOM file and extract pixel data
    dicom = pydicom.dcmread(os.path.join(input_path, file))
    pixel_data = dicom.pixel_array    # Convert pixel data to Hounsfield Units (HU)
    hu_image = pixel_to_hu(pixel_data, dicom)    # Flip image vertically
    flipped_image = flip_vertical(hu_image)    # Scale pixel values to 0-255 and convert to 8-bit unsigned integer
    scaled_image = np.interp(flipped_image, (flipped_image.min(), flipped_image.max()), (0, 255)).astype(np.uint8)    # Save image as high-quality JPG file
    output_file = 'raw%.4d.jpg' % i
    cv2.imwrite(os.path.join(output_path, output_file), scaled_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    i = i +1


