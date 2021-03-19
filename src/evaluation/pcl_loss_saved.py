###############################################################################
# Calculate loss for saved point clouds with GT point clouds
# Disctance metrics - chamfer; TODO : add emd
###############################################################################

import os, sys
from os.path import join
import argparse
import glob
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../chamfer_utils')

import cv2
import numpy as np
import scipy.misc as sc
import tensorflow as tf

import tf_nndistance
# import pdb
# import show3d_balls
from shapenet_taxonomy import shapenet_category_to_id
# from helper_funcs import create_folder, save_screenshots, remove_outliers
from helper_funcs import create_folder, remove_outliers


parser = argparse.ArgumentParser()

parser.add_argument('--exp', type=str, required=True,
        help='Name of the experiment')
parser.add_argument('--gpu', type=int, default=0,
        help='gpu id to be used')
parser.add_argument('--calc_metrics', action='store_true',
        help='calculate and save the metrics')
parser.add_argument('--save_outputs', action='store_true',
        help='save the output point cloud projections from different views')
parser.add_argument('--display', action='store_true',
        help='display the outputs for each input')
parser.add_argument('--rotate', action='store_true',
        help='rotate predicted pcl before metric calculation')
parser.add_argument('--ITER', type=int, required=True,
        help='Iteration number of saved model')
parser.add_argument('--batch_size', type=int, default=8,
        help='batch size to be used')
parser.add_argument('--N_PTS', type=int, default=1024,
        help='Number of points in point cloud')
parser.add_argument('--use_custom_gt', action='store_true', help='Flag to decide which ground truth pointloud to use for evaluation.')

FLAGS = parser.parse_args()

categs = ['chair']
modes = ['test']

#categs = ['airplane', 'car', 'chair']
#modes = ['test', 'train', 'val']


def rotate(xyz, xangle=0, yangle=0, inverse=False):
    xangle = xangle*np.pi/180.
    yangle = yangle*np.pi/180.
    rotmat = np.eye(3)

    rotmat=rotmat.dot(np.array([
	    [1.0,0.0,0.0],
	    [0.0,np.cos(xangle),-np.sin(xangle)],
	    [0.0,np.sin(xangle),np.cos(xangle)],
	    ]))

    rotmat=rotmat.dot(np.array([
	    [np.cos(yangle),0.0,-np.sin(yangle)],
	    [0.0,1.0,0.0],
	    [np.sin(yangle),0.0,np.cos(yangle)],
	    ]))

    if inverse:
	    rotmat = np.linalg.inv(rotmat)

    return xyz.dot(rotmat)


def scale(gt_pc, pr_pc): #pr->[-1,1], gt->[-1,1]
    # Scale gt and prediced point clouds to [-1, 1] cube for calculating
    # metrics
    pred = tf.cast(pr_pc, dtype=tf.float32)
    gt   = tf.cast(gt_pc, dtype=tf.float32)

    pred_clean = tf.clip_by_value(pred,-0.5,0.5)

    min_gt = tf.convert_to_tensor([tf.reduce_min(gt[:,:,i], axis=1) for i in xrange(3)])
    max_gt = tf.convert_to_tensor([tf.reduce_max(gt[:,:,i], axis=1) for i in xrange(3)])

    min_pr = tf.convert_to_tensor([tf.reduce_min(pred_clean[:,:,i], axis=1) for i in xrange(3)])
    max_pr = tf.convert_to_tensor([tf.reduce_max(pred_clean[:,:,i], axis=1) for i in xrange(3)])

    length_gt = tf.abs(max_gt - min_gt)
    length_pr = tf.abs(max_pr - min_pr)

    diff_gt = tf.reduce_max(length_gt, axis=0, keep_dims=True) - length_gt
    diff_pr = tf.reduce_max(length_pr, axis=0, keep_dims=True) - length_pr

    new_min_gt = tf.convert_to_tensor([min_gt[i,:] - diff_gt[i,:]/2. for i in xrange(3)])
    new_max_gt = tf.convert_to_tensor([max_gt[i,:] + diff_gt[i,:]/2. for i in xrange(3)])
    new_min_pr = tf.convert_to_tensor([min_pr[i,:] - diff_pr[i,:]/2. for i in xrange(3)])
    new_max_pr = tf.convert_to_tensor([max_pr[i,:] + diff_pr[i,:]/2. for i in xrange(3)])

    size_pr = tf.reduce_max(length_pr, axis=0)
    size_gt = tf.reduce_max(length_gt, axis=0)

    scaling_factor_gt = 1. / size_gt # 2. is the length of the [-1,1] cube
    scaling_factor_pr = 1. / size_pr

    box_min = tf.ones_like(new_min_gt) * -0.5

    adjustment_factor_gt = box_min - scaling_factor_gt * new_min_gt
    adjustment_factor_pr = box_min - scaling_factor_pr * new_min_pr

    pred_scaled = tf.transpose((tf.transpose(pred) * scaling_factor_pr)) + tf.reshape(tf.transpose(adjustment_factor_pr), (-1,1,3))
    gt_scaled   = tf.transpose((tf.transpose(gt) * scaling_factor_gt)) + tf.reshape(tf.transpose(adjustment_factor_gt), (-1,1,3))
    return gt_scaled, pred_scaled


def get_chamfer_metrics(gt_pcl, pred_pcl):
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
    dists_forward = tf.reduce_mean(dists_forward, axis=1) # (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)
    dists_backward = tf.reduce_mean(dists_backward, axis=1)
    chamfer_distance = dists_backward + dists_forward
    return dists_forward, dists_backward, chamfer_distance


def get_feed_dict(cnt, batch_size, models):
    gt = []; pred = []; n_err=0;
    for i in range(batch_size):
        try:
            model = models[cnt+i]
            pred_pcl = np.load(model).astype(np.float32)
            model_name = model.split('/')[-1].split('_')[0]
            gt_pcl_filename = 'pointcloud_1024.npy' if not FLAGS.use_custom_gt else 'gt_pointcloud_1024.npy'
            gt_pcl = np.load(join(gt_pcl_dir, model_name, gt_pcl_filename)).astype(np.float32)
            if FLAGS.rotate:
                pred_pcl = rotate(rotate(pred_pcl[:,:3], 0, 90), 90, 0)

            # print("dimensions of pred_pcl", pred_pcl.shape)
            # print("dimensions of gt_pcl", gt_pcl.shape)

            gt_pcl_subset = gt_pcl[:,:3]
            pred_pcl_subset = pred_pcl[:,:3]

            # print("dimensions of gt_pcl_subset", gt_pcl_subset.shape)
            # print("dimensions of pred_pcl_subset", pred_pcl_subset.shape)

            gt.append(gt_pcl_subset)
            pred.append(pred_pcl_subset)

	except KeyboardInterrupt:
	    sys.exit()
        except:
            gt.append(np.ones((1024, 3)))
            pred.append(np.ones((1024, 3)))
            n_err += 1
    feed_dict = {pcl_gt: gt, pcl_pred: pred}
    return feed_dict, n_err


pcl_gt = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.N_PTS, 3),
        name='gt_pcl')
pcl_pred = tf.placeholder(tf.float32, (FLAGS.batch_size, FLAGS.N_PTS, 3),
        name='pred_pcl')
pcl_gt_scaled, pcl_out_scaled = scale(pcl_gt, pcl_pred)
dists_forward_scaled, dists_backward_scaled, chamfer_distance_scaled = get_chamfer_metrics(
                                                pcl_gt_scaled, pcl_out_scaled)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

fwd_np_arr = []
bwd_np_arr = []
chamfer_np_arr = []

with tf.Session(config=config) as sess:
    for mode in modes:
        for categ in categs:
            # out_dir = join(FLAGS.exp, 'screenshots')
            out_fwd_path = join(FLAGS.exp, 'FWD.npy' if not FLAGS.use_custom_gt else "FWD_gt.npy")
            out_bwd_path = join(FLAGS.exp, 'BWD.npy' if not FLAGS.use_custom_gt else "BWD_gt.npy")
            out_chamfer_path = join(FLAGS.exp, 'CMF.npy' if not FLAGS.use_custom_gt else "CMF_gt.npy")
            # create_folder([out_dir])
            categ_id = shapenet_category_to_id[categ]
            N_ERR = 0
            fwd_dist = 0.; bwd_dist = 0.; chamfer_dist = 0.;
            pred_dir = join(FLAGS.exp, 'log_proj_pcl_%s_rot'%mode)
            # print("pred_dir is", pred_dir)
            gt_pcl_dir = '../../data/ShapeNet_v1/%s'%(categ_id)
            # print("gt_pcl_dir is", gt_pcl_dir)
            img_data_dir = '../../data/ShapeNet_rendered/%s'%(categ_id)
            # print("img_data_dir is", img_data_dir)
            img_models = sorted(np.load('../../splits/images_list_%s_%s.npy'%(categ_id, mode), allow_pickle=True))

            models = sorted(glob.glob(join(pred_dir, '*_pred.npy')))
            # print("length of models is", len(models))


            count = 0

            for cnt in range(len(models)//FLAGS.batch_size):
                if cnt%100==0:
                    print cnt, '/', len(models)//FLAGS.batch_size

                model_name, model_id = img_models[cnt][0].split('_')
                ip_img = cv2.imread(join(img_data_dir, model_name,
                    'render_%s.png'%model_id))

                feed_dict, n_err = get_feed_dict(cnt, FLAGS.batch_size, models)
                fwd, bwd, chamfer = sess.run([dists_forward_scaled,
                        dists_backward_scaled, chamfer_distance_scaled],
                        feed_dict)
                _pcl_gt, _pcl_out = sess.run([pcl_gt_scaled, pcl_out_scaled],
                        feed_dict)
                # pdb.set_trace()

                N_ERR += n_err
                fwd_dist += np.mean(fwd)
                bwd_dist += np.mean(bwd)
                chamfer_dist += np.mean(chamfer)

                # Add to list
                fwd_np_arr.append(fwd)
                bwd_np_arr.append(bwd)
                chamfer_np_arr.append(chamfer)

                if FLAGS.display:
                    print("disabling display option in VM")
                    # cv2.imshow('img', ip_img)
                    # _pcl_gt[0] = rotate(_pcl_gt[0], 90, 90)
                    # _pcl_out[0] = rotate(_pcl_out[0], 90, 90)
                    # show3d_balls.showpoints(_pcl_gt[0], ballradius=3)
                    # show3d_balls.showpoints(_pcl_out[0], ballradius=3)
                    # saveBool = show3d_balls.showtwopoints(_pcl_gt[0], _pcl_out[0], ballradius=3)
                    print 'Model:%s, Ch:%.5f, fwd:%.5f, bwd:%.5f'%(model_name, chamfer, fwd, bwd)

                elif FLAGS.save_outputs:
                    gt_rot = rotate(_pcl_gt[0], 90)
                    pred_rot = rotate(_pcl_out[0], 90)
                    # save_screenshots(gt_rot, pred_rot, ip_img,
                    #         out_dir, model_name + '_' + model_id, mode)

                count = cnt

            ## TODO: double-check if this denominator is correct
            fwd_dist = (fwd_dist / count) * 1000
            bwd_dist = (bwd_dist / count) * 1000
            chamfer_dist = (chamfer_dist / count) * 1000

            # Save files to output dir from np arrays.
            np.save(out_fwd_path, np.array(fwd_dist), allow_pickle=True)
            np.save(out_bwd_path, np.array(bwd_dist), allow_pickle=True)
            np.save(out_chamfer_path, np.array(chamfer_dist), allow_pickle=True)


            print 'N_ERR', N_ERR
            print 'Categ: %s, Mode: %s, Chamfer: , Fwd:  Bwd: %.3f, %.3f, %.3f'%(categ, mode, chamfer_dist, fwd_dist, bwd_dist)
