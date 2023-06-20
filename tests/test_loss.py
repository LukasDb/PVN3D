from PVN3D.losses.pvn_loss import PvnLoss
import tensorflow as tf
import numpy as np


def _get_cube_keypoints(h, w, t):
    keypoints = np.array(
        [
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )

    # Scale the keypoints by the height, width, and thickness
    keypoints[:, 0] *= h
    keypoints[:, 1] *= w
    keypoints[:, 2] *= t
    return keypoints


def test_get_offst_centerpoint():
    # get_offst transforms the offsets between a poincloud and a set of keypoints, transform by RT
    # RT is the transformation matrix from the object frame to the camera frame
    # pcld_xyz is the pointcloud in the camera frame
    # mask_selected is a mask of the points in the pointcloud that belong to the object
    # kpts_cpts is the keypoints in the object frame
    # kp_offsets is the offsets between the keypoints and the pointcloud
    # cp_offsets is the offsets between the centerpoint and the pointcloud

    # Define the keypoints of the cube
    kpts_cpts = _get_cube_keypoints(1.0, 1.0, 1.0)

    RT = np.eye(4)
    RT[2, 3] = 1.0  # cube is 1m from camera, centered
    # this means the pointcloud is plane with the front of the cube

    # mock pointcloud as a 1x1m grid of points, 0.5m away from camera
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3))
    pcld_xyz = np.zeros((9, 3))
    pcld_xyz[:, 0] = [-0.5, 0.0, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5]
    pcld_xyz[:, 1] = [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    pcld_xyz[:, 2] = 1.0  # [9,3]
    print("Testing with pointcloud:\n", pcld_xyz)

    # mock mask_selected as all points in the pointcloud
    mask_selected = np.ones((9, 1))

    # mock offsets as the difference between the keypoints and the pointcloud
    offsets_cp_gt = np.array(
        [
            [0.5, 0.5, 0.0],
            [0, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, 0., 0.0],
            [0, 0., 0.0],
            [-0.5, 0., 0.0],
            [0.5, -0.5, 0.0],
            [0., -0.5, 0.0],
            [-0.5, -0.5, 0.0],
        ]
    )  # [9, 3]
    offsets_cp_gt = offsets_cp_gt[None, :, None, :]  # [1, 9, 1, 3]

    kp_off, cp_off = PvnLoss.get_offst(
        RT[None, ...],
        pcld_xyz[None, ...],
        mask_selected[None, ...],
        kpts_cpts[None, ...],
    )
    cp_off = cp_off.numpy()
    print("Got cp_off:\n", cp_off)
    print("Expected:\n", offsets_cp_gt)
    assert cp_off.shape == (1, 9, 1, 3)
    assert np.all((cp_off - offsets_cp_gt) < 1e-6)


def test_l1loss_zero_loss():
    """test loss==0 when offset_pred==offset_gt"""
    bs = 5
    n_pts = 4096
    n_kpts = 8
    offset_pred = np.random.uniform(size=(bs, n_pts, n_kpts, 3))
    offset_pred = tf.constant(offset_pred, dtype=tf.float32)
    offset_gt = tf.constant(offset_pred, dtype=tf.float32)
    mask_labels = tf.ones((bs, n_pts), dtype=tf.int32)
    assert PvnLoss.l1_loss(offset_pred, offset_gt, mask_labels) == tf.constant(0.0)


def test_l1loss_masked_zero_loss():
    """test loss==0 when offset_pred==offset_gt"""
    bs = 1
    n_pts = 10
    n_kpts = 8
    offset_pred = np.random.uniform(size=(bs, n_pts, n_kpts, 3))
    mask_labels = np.ones((bs, n_pts))

    # mask out 3 points and offset them (so they are "wrong")
    mask_labels[0, 0] = 0
    mask_labels[0, 1] = 0
    mask_labels[0, 2] = 0
    offset_gt = offset_pred.copy()
    offset_gt[0, 0] += 0.2
    offset_gt[0, 1] -= 0.2
    offset_gt[0, 2] += 0.8

    offset_pred = tf.constant(offset_pred, dtype=tf.float32)
    offset_gt = tf.constant(offset_gt, dtype=tf.float32)
    mask_labels = tf.constant(mask_labels, dtype=tf.int32)
    assert PvnLoss.l1_loss(offset_pred, offset_gt, mask_labels) == tf.constant(0.0)


def test_l1loss_set_loss():
    """test loss==0 when offset_pred==offset_gt"""
    bs = 3
    n_pts = 10
    n_kpts = 8
    error = 0.2

    offset_pred = np.ones((bs, n_pts, n_kpts, 3))
    offset_gt = offset_pred.copy()
    offset_gt += error
    mask_labels = np.ones((bs, n_pts))

    offset_pred = tf.constant(offset_pred, dtype=tf.float32)
    offset_gt = tf.constant(offset_gt, dtype=tf.float32)
    mask_labels = tf.constant(mask_labels, dtype=tf.int32)
    expected_l1_distance = (
        error * n_kpts * 3
    )  # summed manhattan distance over keypoints
    assert (
        PvnLoss.l1_loss(offset_pred, offset_gt, mask_labels)
        - tf.constant(expected_l1_distance, dtype=tf.float32)
        < 1e-6
    )
