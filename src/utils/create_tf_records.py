#########################################################
# Create tensorflow records file for (projected) images
#########################################################

import os, sys
from os.path import join, abspath, basename
import glob
import time
import pdb

import tensorflow as tf
import numpy as np


######## Imports for masking
import skimage.filters
import skimage.io

from PIL import  Image
###########

# Change these values based on the data for which tfrecords file needs to be
# created (do them sequentially - set only one of them to True at a time).
save_img = False
save_pcl = False
save_pose = True
# Jaineel Dalal: Adding save_mask as an option over here. It doesn't exist in original code.
# Note: Make sure to run create_tf_records for mask right before image. This will generate all missing masks.
save_mask = False

dataset = 'shapenet'
if save_img or save_mask or save_pose:
    data_dir = '../../data/ShapeNet_rendered'
elif save_pcl:
    data_dir = '../../data/ShapeNet_v1'
out_dir = '../../data'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

categs = ['02691156', '02958343', '03001627']
modes = ['train', 'val', 'test']


def read_img_or_mask(img_path, image, mask, cnt=None):
    # Read only one of the projections for each model. Randomly choose the one
    # out of 10 existing projections
    # if os.path.exists(img_path):
    #     print("path exists")
    if cnt==None:
        cnt = np.random.randint(0,10)
    fullImgPath = join(img_path, 'render_%s.png'%cnt)
    fullMaskPath = join(img_path, 'mask_%s.png'%cnt)
    if image:
        # The corresponding mask for this image does not exist. Do not add this image.
        if not os.path.exists(fullMaskPath):
            # print('not adding ', fullMaskPath)
            # try opening the fullMaskPath to trigger an exception.
            open(fullMaskPath)
            # return
        with open(fullImgPath) as f:
            img_bytes = f.read()
    elif mask:
        # The corresponding image for this mask does not exist. Do not add this mask.
        if not os.path.exists(fullImgPath):
            # print('not adding ', fullImgPath)
            # try opening the fullImgPath to trigger an exception.
            open(fullImgPath)
            # return
        # If the mask does not exist, generate one and store at the path before reading it.
        if not os.path.exists(fullMaskPath):
            img = skimage.io.imread(fname=fullImgPath)
            val = skimage.filters.threshold_otsu(img)
            img_bool = img >= val
            maxval = 255
            img_bin = img_bool * maxval
            Image.fromarray(np.uint8(img_bin)).save(fullMaskPath)
        with open(fullMaskPath) as f:
            img_bytes = f.read()

    img_name = img_path.split('/')[-1]+'_%s'%cnt
    # print("img_name is %s"%img_name)

    fileNameInput = None
    imageInput = None
    featureInput = None
    example = None
#    print(img_path)
    try:
#	print('block 1')
        fileNameInput = tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_name]))
        imageInput = tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_bytes]))
    except Exception as expt:
        print("an exception occurred while instantiating fileNameInput and imageInput")
        print("exception string is %s"%expt)

    try:
# 	print('block 2')
        featureInput = tf.train.Features(feature = {'filename': fileNameInput, 'image': imageInput})
    except Exception as expt:
        print("an exception occurred while instantiating featureInput")
        print("exception string is %s"%expt)

    try:
#	print('block 3')
        ## this line is the buggy line
        ## example = tf.train.Example(features = tf.train.Features(feature = {featureInput}))
        examples = tf.train.Example(features = featureInput)
    	example_str = examples.SerializeToString()
 #   	print('Example string: ', examples)
    	return example_str
    except Exception as expt:
        print("an exception occurred while instantiating example")
        print("exception string is %s"%expt)
 	return None
   # example_str = example.SerializeToString()
   # print('Example string: ', example_str)
   # return example_str


def read_pose(img_path, cnt=None):
    # Read only one of the projections for each model. Randomly choose one
    # out of 10 existing projections or use the provided id
    if cnt==None:
        cnt = np.random.randint(0,10)
    if save_pose:
	print(img_path)
         # Jaineel Dalal: There is no view.txt file at the given path! How do I generate poses?
    	with open(join(img_path, 'view.txt'), 'r') as fp:
	   angles = [item.split('\n')[0] for item in fp.readlines()]
    print('Ok: image opened at', img_path)
    angle = angles[int(cnt)]
    angle = [float(item)*(np.pi/180.) for item in angle.split(' ')[:2]]
    pose_bytes = np.array(angle).tostring()

    img_name = img_path.split('/')[-1]+'_%s'%cnt
    print(img_name)
    example = tf.train.Example(features = tf.train.Features(feature = { \
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name])),
        'pose': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_bytes]))}))
    example_str = example.SerializeToString()
    print(example_str)
    return example_str


def read_pcl(pcl_path):
    # print(pcl_path)
    # Jaineel Dalal: this does not work. This file does not exist.
    # Our two options are: pointcloud_1024.npy  pointcloud_2048.npy
    #pcl = np.load(join(pcl_path, 'pcl_1024_fps_trimesh_colors.npy'))
    pcl = np.load(join(pcl_path, 'pointcloud_1024.npy'))
    pcl_bytes = pcl.tostring()
    pcl_name = pcl_path.split('/')[-1]
    example = tf.train.Example(features = tf.train.Features(feature = { \
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_name])),
        'pcl': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pcl_bytes]))}))
    example_str = example.SerializeToString()
    return example_str


for mode in modes:
    for categ in categs:
        print categ
        overwrite = True

        ###########
        # Experimental code based on a different splits.txt file inside the data/ directory

        # Get models_only ids.
        # file_name = '%s_%s_list.txt'%(categ, mode)
        # model_file_path = join(data_dir, 'splits', file_name)
        # print(model_file_path, ' is the model path.')
        # with open(model_file_path) as f:
        #     models_only = [line.rstrip() for line in f]

        # # Go inside each models file and get the images for each render_image_id.png
        # for model_name in models_only:
        #     model_path = join(data_dir, categ, model_name)
        #     filelist=os.listdir(model_path)
        #     for file in filelist:

        ########
        model_path = '../../splits/images_list_%s_%s.npy'%(categ, mode)
        models = np.load(model_path, allow_pickle=True)
        image_ids = [model[0].split('_')[1] for model in models]
        models = [model[0].split('_')[0] for model in models]
        if save_img:
            if dataset == 'pfcn':
                out_file = '%s_%s_image_pfcn.tfrecords'%(join(out_dir, categ),
                        mode)
            else:
                out_file = '%s_%s_image.tfrecords'%(join(out_dir, categ), mode)
        elif save_pcl:
            out_file = '%s_%s_pcl_color.tfrecords'%(join(out_dir, categ), mode)
        elif save_pose:
            out_file = '%s_%s_pose.tfrecords'%(join(out_dir, categ), mode)
        elif save_mask:
            out_file = '%s_%s_mask.tfrecords'%(join(out_dir, categ), mode)

        if os.path.exists(out_file):
            response = raw_input('File exists! Replace? (y/n): ')
            if response != 'y':
                overwrite = False

        N_missing = 0
        time_st = time.time()
        if overwrite:
            with tf.python_io.TFRecordWriter(out_file) as writer:
                for idx, model in enumerate(models):
                    if idx%500==0:
                        print idx, '/', len(models)
                        print 'Time: ', (time.time() - time_st)//60
                    try:
                        if save_img and save_mask:
                            print('Error only one of image or mask can be specified. Rerun with one option.')
                            sys.exit()
                        elif save_img or save_mask:
                            img_or_mask_str = read_img_or_mask(join(data_dir, categ, model), save_img, save_mask,
                                    image_ids[idx])
                            writer.write(img_or_mask_str)
                        elif save_pose:
                            fullPath = join(data_dir, categ, model)
                            pose_str = read_pose(join(data_dir, categ, model),
                                    image_ids[idx])
                            writer.write(pose_str)
                        elif save_pcl:
                            pcl_str = read_pcl(join(data_dir, categ, model))
                            writer.write(pcl_str)
                    except KeyboardInterrupt:
                        sys.exit()
                    except Exception as expt:
                        # print("system error is", expt)
                        N_missing += 1
                        continue
            print 'Time: ', (time.time() - time_st)/60
            print 'N_missing: ', N_missing
