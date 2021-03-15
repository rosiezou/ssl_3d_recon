##############################################################################
#                       Calculate loss values
##############################################################################

import numpy as np
from scipy.spatial.distance import cdist as np_cdist
import tensorflow as tf

from chamfer_utils import tf_nndistance
from chamfer_utils import tf_auctionmatch

def get_img_loss(gt, pred, mode='l2_sq', affinity_loss=False):
    '''
    Loss in 2D domain - on mask or rgb images.
    Args:
        gt: (BS, H, W, *); ground truth mask or rgb
        pred: (BS, H, W, *); predicted mask or rgb
        mode: str; loss mode
        affinity_loss: boolean; affinity loss will be added to mask loss if
                                set to True
    Returns:
        loss: (); averaged loss value
        min_dist: (); averaged forward affinity distance
        min_dist_inv: (); averaged backward affinity distance
    '''
    grid_h, grid_w = 64, 64
    dist_mat = grid_dist(grid_h, grid_w)
    min_dist = tf.zeros(()); min_dist_inv = tf.zeros(());
    if mode=='l2_sq':
        loss = (gt - pred)**2
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    elif mode=='l2':
        loss = (gt - pred)**2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(loss)
    if mode=='l1':
        loss = tf.abs(gt - pred)
        loss = tf.reduce_mean(loss)
    elif mode == 'bce':
	print '\nBCE Logits Loss\n'
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=pred)
        loss = tf.reduce_mean(loss)
    elif mode == 'bce_prob':
	print '\nBCE Loss\n'
        epsilon = 1e-8
        loss = -gt*tf.log(pred+epsilon) - (1-gt)*tf.log(tf.abs(1-pred-epsilon))
        loss = tf.reduce_mean(loss)
    if affinity_loss:
	dist_mat += 1.
	gt_mask = gt #+ (1.-gt)*1e6*tf.ones_like(gt)
	gt_white = tf.expand_dims(tf.expand_dims(gt,3),3)
	gt_white = tf.tile(gt_white, [1,1,1,grid_h,grid_w])

	pred_white = tf.expand_dims(tf.expand_dims(pred,3),3)
	pred_white = tf.tile(pred_white, [1,1,1,grid_h,grid_w])

	gt_white_th = gt_white + (1.-gt_white)*1e6*tf.ones_like(gt_white)
	dist_masked = gt_white_th * dist_mat * pred_white

	pred_mask = (pred_white) + ((1.-pred_white))*1e6*tf.ones_like(pred_white)
	dist_masked_inv = pred_mask * dist_mat * gt_white

	min_dist = tf.reduce_mean(tf.reduce_min(dist_masked, axis=[3,4]))
	min_dist_inv = tf.reduce_mean(tf.reduce_min(dist_masked_inv, axis=[3,4]))
    return loss, min_dist, min_dist_inv


## helper functions for k-random-octant loss
def compute_l2_norm(pcl_mat, octants):
    # pcl_mat: BATCH_SIZE x 1024 x 3 tensor
    # octants: BATCH_SIZE k x 3 tensor
    # output: BATCH_SIZE x 1024 x k matrix where each row contains l2 norm of each point against each of the k centers.
    
    ## this is inefficient for larger batches
    num_points = pcl_mat.shape[1]
    num_octants = octants.shape[1]
    distance = np.zeros((num_points, num_octants))
    num_batches = pcl_mat.shape[0]
    outputTensor = tf.zeros(shape = (1, pcl_mat.shape[1], octants.shape[1]))
    for batch in range(num_batches):
        output_batch = tf.zeros(shape = (1, pcl_mat.shape[1], 0))
        for octant in range(num_octants):
            # Tensor("ExpandDims_62:0", shape=(1, 1, 3), dtype=float32)
            # Tensor("sub_60:0", shape=(1, 1024, 3), dtype=float32)
            # Tensor("norm/Sqrt:0", shape=(1, 1024, 1), dtype=float32)
            single_octant_center = tf.expand_dims(
                tf.expand_dims(octants[batch, octant, :], 
                    axis = 0), axis = 0)
            diff = pcl_mat[batch, :, :] - single_octant_center
            norm_for_all_points = tf.norm(diff, axis = 2, keepdims = True)
            output_batch = tf.concat([output_batch, norm_for_all_points], axis = 2)
        outputTensor = tf.concat([outputTensor, output_batch], axis = 0)
    outputTensor = outputTensor[1:, :, :]
    return outputTensor

def find_min_distance_to_cluster(distance, pcl):
    # distance: 1024 x k matrix where each row contains l2 norm of each point against each of the k centers.
    # pcl: 1024 x 3 point clouds
    # output: Tuple of a 1xk matrix contains the row index of the min entry and another
    # 1 x k matrix where each entry contains the min_distance from a given pointcloud point to the
    # kth cluster.
    min_distance = tf.reduce_min(distance, axis=1, keepdims=True)
    min_distance_row_index = tf.argmin(distance, axis = 1)
    
    ## gather points for first batch
    anchor_points = tf.gather(pcl[0, :, :], min_distance_row_index[0, :])
    anchor_points = tf.expand_dims(anchor_points, axis = 0)

    ## stack the rest
    for i in range(1, pcl.shape[0]):
        anchor_points = tf.concat([anchor_points, tf.expand_dims(tf.gather(pcl[i, :, :], min_distance_row_index[i, :]), axis = 0)], axis = 0)

    return anchor_points, min_distance_row_index

## TODO: add checks for octants not containing any anchor points
##       should those be omitted? or still summed up but then we
##       add the rigidity score as a regularization term?

def getAnchorPoints(pcl_tensor, k):
    '''
    pcl_tensor: tensor of dim = (BS,N_pts,3)
    '''
    min_vals = tf.reduce_min(pcl_tensor, axis = 1)
    max_vals = tf.reduce_max(pcl_tensor, axis = 1)
    intervals = (max_vals - min_vals)/k
    
    ## this works but it's inefficient for large values of k
    batchArray = []
    for batch in range(pcl_tensor.shape[0]):
        x_min, y_min, z_min = min_vals[batch, 0], min_vals[batch, 1], min_vals[batch, 2]
        x_max, y_max, z_max = max_vals[batch, 0], max_vals[batch, 1], max_vals[batch, 2]
        x_interval, y_interval, z_interval = intervals[batch, 0], intervals[batch, 1], intervals[batch, 2]
        ## compute geometric centers
        searchVolumes = []
        overallIndex = 1
        x_init, y_init, z_init = x_min - x_interval/2, y_min - y_interval/2, z_min - z_interval/2
        for p in range(0, k):
            x_init += x_interval
            for q in range(0, k):
                y_init += y_interval
                for r in range(0, k):
                    z_init += z_interval
                    searchVolumes.append([x_init, y_init, z_init])
                    overallIndex += 1
                z_init = z_min - z_interval/2
            y_init = y_min - y_interval/2
                
        batchArray.append(searchVolumes)
    octants = tf.convert_to_tensor(batchArray)

    ## get anchor points' distances
    ## L2_norm([1024, 3], [27, 3]) ==> [1024, 27] all point clouds dist to all geometric centers ==> [1, 27]
    distance = compute_l2_norm(pcl_tensor, octants)
    return find_min_distance_to_cluster(distance, pcl_tensor)

def getCenterPoint(pcl_tensor):
    min_vals = tf.reduce_min(pcl_tensor, axis = 1)
    max_vals = tf.reduce_max(pcl_tensor, axis = 1)
    geometric_center = tf.expand_dims(min_vals + (max_vals - min_vals)/2, axis = 1)
    l2_norms = compute_l2_norm(pcl_tensor, geometric_center)
    return find_min_distance_to_cluster(l2_norms, pcl_tensor)

def get_k_random_octant_loss(gt, pred, min_k, max_k):
    '''
    K random octant loss
    Args:
        gt: (BS,N_pts,3); GT point cloud
        pred: (BS,N_pts,3); predicted point cloud
        min_k: min value for uniformly sampled k
        max_k: max value for uniformly sampled k
    Returns:
        loss: (); averaged chamfer/emd loss
    Note:
        k = random number to divide each axis
        total # of octants = k^3
    '''
    ## fetch all anchor points
    k = int(round(np.random.uniform(min_k, max_k)))
    gt_anchor_points, _ = getAnchorPoints(gt, k)
    pred_anchor_points, _ = getAnchorPoints(pred, k)

    ## fetch the center points
    gt_center_point, _= getCenterPoint(gt)
    pred_center_point, _ = getCenterPoint(pred)

    ## calculate sum of Euclidean distance between anchor points and center points, for both gt and pred
    gt_anchor_to_center_dist = tf.norm(gt_anchor_points - gt_center_point, axis = 2)
    pred_anchor_to_center_dist = tf.norm(pred_anchor_points - pred_center_point, axis = 2)

    ## return difference of the two sums of Euclidean distance
    ##    currently all octants are considered, regardless if they're empty
    loss = tf.reduce_sum(gt_anchor_to_center_dist, axis = 1) - tf.reduce_sum(pred_anchor_to_center_dist, axis = 1)
    loss = tf.reduce_mean(loss, axis = 0)
    return loss

def get_2d_symm_loss(img):
    '''
    Symmetry loss: to enforce symmetry in image along vertical axis in the
    image
    Args:
        img: (BS,H,W,3); input image
    Returns:
        loss: (); averaged symmetry loss, L1
    '''
    W = img.get_shape().as_list()[2]
    loss = tf.abs(img[:,:,:W/2,:]-img[:,:,W/2:,:])
    loss = tf.reduce_mean(loss)
    return loss

def get_3d_loss(gt, pred, mode='chamfer'):
    '''
    3D loss: to enforce 3D consistency loss
    Args:
        gt: (BS,N_pts,3); GT point cloud
        pred: (BS,N_pts,3); predicted point cloud
        mode: str; method to calculate loss - 'chamfer' or 'emd'
    Returns:
        loss: (); averaged chamfer/emd loss
    '''
    if mode=='chamfer':
        _, _, loss = get_chamfer_dist(gt, pred)
    elif mode=='emd':
        loss = get_emd_dist(gt, pred)
    loss = tf.reduce_mean(loss)
    return loss


def get_pose_loss(gt, pred, mode='l1'):
    '''
    Pose loss: to enforce pose consistency loss
    Args:
        gt: (BS,2); GT pose - azimuth and elevation for each pcl
        pred: (BS,2); predicted pose
        mode: str; method to calculate loss
    Returns:
        loss: (); averaged loss
    '''
    if mode=='l1':
        loss = tf.reduce_mean(tf.abs(gt - pred))
    elif mode=='l2_sq':
        loss = (gt - pred)**2
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    elif mode=='l2':
        loss = (gt - pred)**2
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(loss)
    elif loss == 'cosine_dist':
	print '\nCosine Distance\n'
	pred_norm = tf.sqrt(tf.reduce_sum(pred**2, axis=-1)+1e-8)
	loss = 1. - (tf.reduce_sum(gt*pred, axis=-1)/(pred_norm+1e-8))
    return loss


def get_chamfer_dist(gt_pcl, pred_pcl):
    '''
    Calculate chamfer distance between two point clouds
    Args:
        gt_pcl: (BS,N_pts,3); GT point cloud
        pred_pcl: (BS,N_pts,3); predicted point cloud
    Returns:
        dists_forward: (); averaged forward chamfer distance
        dists_backward: (); averaged backward chamfer distance
        chamfer_distance: (); averaged chamfer distance
    '''
    # FWD: GT-->Pred, Bwd: Pred-->GT
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt_pcl, pred_pcl)
    dists_forward = tf.reduce_mean(dists_forward) # (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)
    dists_backward = tf.reduce_mean(dists_backward)
    chamfer_distance = dists_backward + dists_forward
    return dists_forward, dists_backward, chamfer_distance


def get_emd_dist(gt_pcl, pred_pcl):
    '''
    Calculate emd between two point clouds
    Args:
        gt_pcl: (BS,N_pts,3); GT point cloud
        pred_pcl: (BS,N_pts,3); predicted point cloud
    Returns:
        emd: (BS); averaged emd
    '''
    batch_size, num_points = (gt_pcl.shape)[:2]
    X,_ = tf.meshgrid(range(batch_size), range(num_points), indexing='ij')
    ind, _ = auction_match(pred_pcl, gt_pcl) # Ind corresponds to points in pcl_gt
    ind = tf.stack((X, ind), -1)
    emd = tf.reduce_mean(tf.reduce_sum((tf.gather_nd(gt_pcl, ind) - pred_pcl)**2, axis=-1), axis=1) # (BATCH_SIZE,NUM_POINTS,3) --> (BATCH_SIZE,NUM_POINTS) --> (BATCH_SIZE)
    return emd


def get_partseg_loss(gt, pred, loss='ce_logits'):
    '''
    Calculate part segmentation loss
    Args:
        gt_pcl: (BS,N_pts); GT point cloud
        pred_pcl: (BS,N_pts,N_cls); predicted point cloud
        loss: str; type of loss to be used: 'ce_logits' or 'iou'
    Returns:
        loss: (BS); averaged loss
    '''
    if loss == 'ce_logits':
	print '\nBCE Logits Loss\n'
	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pred)
    elif loss == 'iou':
	print '\nIoU metric\n'
	pred_idx = tf.argmax(pred, axis=3)
	loss = tf.metrics.mean_iou(gt, pred_idx, NUM_CLASSES+1) # tuple of (iou, conf_mat)
    return loss


def grid_dist(grid_h, grid_w):
    '''
    Compute distance between every point in grid to every other point
    '''
    x, y = np.meshgrid(range(grid_h), range(grid_w), indexing='ij')
    grid = np.asarray([[x.flatten()[i],y.flatten()[i]] for i in range(len(x.flatten()))])
    grid_dist = np_cdist(grid,grid)
    grid_dist = np.reshape(grid_dist, [grid_h, grid_w, grid_h, grid_w])
    return grid_dist
