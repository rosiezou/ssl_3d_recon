# Custom script to generate pointcloud data.

import numpy as np
import math
import os
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import random

# Code in this script is derived from:
# 1. https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
# 2. https://github.com/davidstutz/bpy-visualization-utils
# Note this script must be run with python>=3.5


SHAPENET_DATA_DIR = '/data'
PCL_DATA_DIR = '../../data/ShapeNet_v1/'
GT_POINTCLOUD_NAME = 'gt_pointcloud_1024.npy'
INPUT_OBJ_FILE_NAME = 'model.obj'
INTERMEDIATE_OFF_FILE_NAME = 'model.off'

categs = ['03001627']
# categs = ['02691156', '02958343', '03001627']

# Python 3.5 doesn't support random.choices.
# This function is meant to be a replacement for the same.
def choices(population, weights=None, cum_weights=None, k=1):
    """Return a k sized list of population elements chosen with replacement.
    If the relative weights or cumulative weights are not specified,
    the selections are made with equal probability.
    """
    n = len(population)
    if cum_weights is None:
        if weights is None:
            _int = int
            n += 0.0    # convert to float for a small speed improvement
            return [population[_int(random.random() * n)] for i in _repeat(None, k)]
        cum_weights = list(_accumulate(weights))
    elif weights is not None:
        raise TypeError('Cannot specify both weights and cumulative weights')
    if len(cum_weights) != n:
        raise ValueError('The number of weights does not match the population')
    bisect = _bisect
    total = cum_weights[-1] + 0.0   # convert to float
    hi = n - 1
    return [population[bisect(cum_weights, random.random() * total, 0, hi)]
            for i in _repeat(None, k)]


def write_off(file, vertices, faces):
    """
    Writes the given vertices and faces to OFF.
    :param vertices: vertices as tuples of (x, y, z) coordinates
    :type vertices: [(float)]
    :param faces: faces as tuples of (num_vertices, vertex_id_1, vertex_id_2, ...)
    :type faces: [(int)]
    """

    num_vertices = len(vertices)
    num_faces = len(faces)
    assert num_vertices > 0
    assert num_faces > 0
    with open(file, 'w+') as fp:
        fp.write('OFF\n')
        fp.write(str(num_vertices) + ' ' + str(num_faces) + ' 0\n')
        for vertex in vertices:
            assert len(vertex) == 3, 'invalid vertex with %d dimensions found (%s)' % (len(vertex), file)
            fp.write(str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')
        for face in faces:
            assert face[0] == 3, 'only triangular faces supported (%s)' % file
            assert len(face) == 4, 'faces need to have 3 vertices, but found %d (%s)' % (len(face), file)
            for i in range(len(face)):
                assert face[i] >= 0 and face[i] < num_vertices, 'invalid vertex index %d (of %d vertices) (%s)' % (face[i], num_vertices, file)
                fp.write(str(face[i]))
                if i < len(face) - 1:
                    fp.write(' ')
            fp.write('\n')
        # add empty line to be sure
        fp.write('\n')


def read_obj(file):
    """
    Reads vertices and faces from an obj file.
    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file
    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        vertices = []
        faces = []
        for line in lines:
            parts = line.split(' ')
            parts = [part.strip() for part in parts if part]
            if parts[0] == 'v':
                assert len(parts) == 4, \
                    'vertex should be of the form v x y z, but found %d parts instead (%s)' % (len(parts), file)
                assert parts[1] != '', 'vertex x coordinate is empty (%s)' % file
                assert parts[2] != '', 'vertex y coordinate is empty (%s)' % file
                assert parts[3] != '', 'vertex z coordinate is empty (%s)' % file
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                assert len(parts) == 4, \
                    'face should be of the form f v1/vt1/vn1 v2/vt2/vn2 v2/vt2/vn2, but found %d parts (%s) instead (%s)' % (len(parts), line, file)
                components = parts[1].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                   'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v1 = int(components[0])
                components = parts[2].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v2 = int(components[0])
                components = parts[3].split('/')
                assert len(components) >= 1 and len(components) <= 3, \
                    'face component should have the forms v, v/vt or v/vt/vn, but found %d components instead (%s)' % (len(components), file)
                assert components[0].strip() != '', \
                    'face component is empty (%s)' % file
                v3 = int(components[0])
                #assert v1 != v2 and v2 != v3 and v3 != v2, 'degenerate face detected: %d %d %d (%s)' % (v1, v2, v3, file)
                if v1 == v2 or v2 == v3 or v1 == v3:
                    # print('[Info] skipping degenerate face in %s' % file)
                    continue
                else:
                    faces.append([v1 - 1, v2 - 1, v3 - 1]) # indices are 1-based!
            else:
              # continue to next line until one of the required args is found.
              continue
        return np.array(vertices, dtype=float), np.array(faces, dtype=int)
    assert False, 'could not open %s' % file


# Convert obj to an off file and save locally
def convert_obj_to_off(obj_file, output_filename):
  if not os.path.exists(obj_file):
    print(obj_file, ' does not exist.')
    return
  obj_vertices, obj_faces = read_obj(obj_file)
  assert obj_vertices.shape[1] == 3
  assert obj_faces.shape[1] == 3
  temp_faces = np.ones((obj_faces.shape[0], 4), dtype = int)*3
  # print(obj_faces.shape)
  temp_faces[:, 1:4] = obj_faces[:, :]
  write_off(output_filename, obj_vertices.tolist(), temp_faces.tolist())


def read_cad_off(filename):
  # Read CAD files with .off extension.
  with open(filename, 'r') as f:
    if 'OFF' != f.readline().strip():
        print('Error: ', filename, ' is a not a file.')
        return false
    num_vertices, num_faces, _ = tuple([int(s) for s in f.readline().strip().split(' ')])
    vertices = [[float(s) for s in f.readline().strip().split(' ')] for ith_vertex in range(num_vertices)]
    faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for ith_face in range(num_faces)]
    return vertices, faces


def calculate_triangle_area(pt1, pt2, pt3):
  # Calculate the area of the triangle formed by the provided points.
  # Note: This area calculation is for a 3D triangle with Heron's formula
  side_a = np.linalg.norm(pt1 - pt2)
  side_b = np.linalg.norm(pt2 - pt3)
  side_c = np.linalg.norm(pt3 - pt1)
  perimeter = 0.5 * ( side_a + side_b + side_c)
  return max(perimeter * (perimeter - side_a) * (perimeter - side_b) * (perimeter - side_c), 0)**0.5


# Sample points on the surface of the chosen triangle
def sample_point_on_triangle(pt1, pt2, pt3):
    # barycentric coordinates on a triangle
    # https://mathworld.wolfram.com/BarycentricCoordinates.html
    # Another good reference: https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1.html
    s, t = sorted([random.random(), random.random()])
    f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
    return (f(0), f(1), f(2))


if __name__ == '__main__':
    for category in categs:
        cat_pcl_data_dir = os.path.join(PCL_DATA_DIR, category)
        print('Category: ', category)
        iteration = 0
        print('Number of files: ', len(os.listdir(cat_pcl_data_dir)))
        for npy_dir in os.listdir(cat_pcl_data_dir):
            iteration+=1
            output_base_dir = os.path.join(cat_pcl_data_dir, npy_dir)
            print('Num: ', iteration)
            if not output_base_dir:
                continue
            # If the output .npy file exists, continue to the next file.
            output_data_path = os.path.join(output_base_dir, GT_POINTCLOUD_NAME)
            if os.path.isfile(output_data_path):
                continue
            input_filepath = os.path.join(SHAPENET_DATA_DIR, category, npy_dir, INPUT_OBJ_FILE_NAME)
            if not os.path.isfile(input_filepath):
                continue
            intermediate_filepath = os.path.join(SHAPENET_DATA_DIR, category, npy_dir, INTERMEDIATE_OFF_FILE_NAME)
            # Convert .obj file to .off file
            print('Converting ', input_filepath, ' to intermediary file: ', intermediate_filepath)
            convert_obj_to_off(input_filepath, intermediate_filepath)
            off_vertices, off_faces = read_cad_off(intermediate_filepath)
            # Calculate area for each face
            areas = np.zeros((len(off_faces)))
            off_vertices = np.array(off_vertices)
            for i in range(len(areas)):
                areas[i] = calculate_triangle_area(off_vertices[off_faces[i][0]], off_vertices[off_faces[i][1]], off_vertices[off_faces[i][2]])
            k = 1024
            sampled_faces = choices(off_faces, weights=areas, k=k)
            # Generate pointcloud.
            print('Generating pointcloud at: ', output_data_path)
            pointcloud = np.zeros((k, 3))
            for i in range(len(sampled_faces)):
                pointcloud[i] = (sample_point_on_triangle(off_vertices[sampled_faces[i][0]], off_vertices[sampled_faces[i][1]], off_vertices[sampled_faces[i][2]]))
            np.save(output_data_path, pointcloud, allow_pickle=True)





