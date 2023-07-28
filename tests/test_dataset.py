from PVN3D.datasets.blender import TrainBlender
import numpy as np
import tensorflow as tf
from numpy.testing import assert_array_equal


train_config = {
    "batch_size": 32,
    "root": "/media/lukas/G-RAID/datasets/blender",
    "data_name": "cps",
    "cls_type": "cpsduck",
    "train_split": 0.9,
    # cutoff: 1000
    "use_cache": True,
    "im_size": [1080, 1920],
}
dataset = TrainBlender(**train_config)
tf_data = dataset.to_tf_dataset()


"""
from .to_tf_dataset

output_signature=(
    (
        tf.TensorSpec(shape=(*self.im_size, 3), dtype=tf.uint8, name="rgb"),
        tf.TensorSpec(shape=(*self.im_size, 1), dtype=tf.float32, name="depth"),
        tf.TensorSpec(shape=(3, 3), dtype=tf.float32, name="intrinsics"),
        tf.TensorSpec(shape=(4,), dtype=tf.int32, name="roi"),
        tf.TensorSpec(shape=(9, 3), dtype=tf.float32, name="mesh_kpts"),
    ),
    (
        tf.TensorSpec(shape=(4, 4), dtype=tf.float32, name="RT"),
        tf.TensorSpec(shape=(*self.im_size, 1), dtype=tf.uint8, name="mask"),
    ),
),

"""


def test_dataloading_shapes():
    for x, y in tf_data:
        b_rgb, b_depth, b_intrinsics, b_roi, b_mesh_kpts = x
        b_RT, b_mask = y
        assert_array_equal(b_rgb.shape, (32, 1080, 1920, 3))
        assert_array_equal(b_depth.shape, (32, 1080, 1920, 1))
        assert_array_equal(b_intrinsics.shape, (32, 3, 3))
        assert_array_equal(b_roi.shape, (32, 4))
        assert_array_equal(b_mesh_kpts.shape, (32, 9, 3))
        assert_array_equal(b_RT.shape, (32, 4, 4))
        assert_array_equal(b_mask.shape, (32, 1080, 1920, 1))

        break


def test_data_values():
    for x, y in tf_data:
        b_rgb, b_depth, b_intrinsics, b_roi, b_mesh_kpts = x
        b_RT, b_mask = y

        assert_array_equal(np.unique(b_mask), [0, 1])
        assert_array_equal(b_mesh_kpts[0], dataset.mesh_kpts.astype(np.float32))

        break
