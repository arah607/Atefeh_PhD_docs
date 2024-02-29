import numpy as np
from csv import writer


def point_inside_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    if np.linalg.norm(np.cross(pt1, pt2)) > 0.0:
        dist = np.linalg.norm(np.cross(vec, (q - pt1))) / np.linalg.norm(vec)
    else:
        print("can't calculate for moment = 0.0")

    return dist <= r


if __name__ == '__main__':

    # contains branch number: {parent, length, radius, angle}
    branch_info = {1: [0, 5.0, 0.5, 45.0], 2: [1, 3.5, 0.4, 35.0], 3: [1, 3.0, 0.25, 36.0], 4: [2, 2.0, 0.15, 45.0],
                   5: [2, 1.5, 0.4, 15.0]}

    # just a demonstration:
    num_pixels = np.array([100, 100, 100])
    resolution = np.array([0.5, 0.5, 0.3])  # define mm/pixel resolution

    with open("coordinate.csv", 'w', newline='') as write_obj:
        for z in range(num_pixels[2]):
            for y in range(num_pixels[1]):
                for x in range(num_pixels[0]):
                    q = np.array([x * resolution[0], y * resolution[1], z * resolution[2]])
                    pt1 = np.array([3.1, 2.6,3.1])  # you will need to calculate pt1 and pt2 from connecting up the tree (from branch_info)
                    pt2 = np.array([3.1, 2.2, 3.2])
                    r = branch_info[1][2]
                    in_cylinder = point_inside_cylinder(pt1, pt2, r, q)

                    csv_writer = writer(write_obj)
                    if in_cylinder:
                        row = [x, y, z]
                        csv_writer.writerow(row)
                        print('point', x, y, z, ' in cylinder')