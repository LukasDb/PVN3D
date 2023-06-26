from PVN3D.models.pvn3d_e2e import PVN3D_E2E
import numpy as np
from numpy.testing import assert_array_equal
import sys

np.set_printoptions(threshold=sys.maxsize)


def test_crop_index():
    """
    crop index gets a integer crop factor and a bounding box with the same aspect ratio
    with batches
    """
    # y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

    resnet_input_shape = [160, 160]  # hxw
    img_h = np.array(1080)
    img_w = np.array(1920)
    roi = np.array(
        [
            [0, 0, 160, 160],  # simplest case
            [0, 0, 320, 320],
            [0, 0, 80, 80],
            [200, 200, 680, 840],  # box with 480x640
            [0, 0, 480, 640],  # same box in top left corner
            [600, 1280, 1080, 1920],  # same box in bottom right corner
        ]
    )
    expected_bbox = np.array(
        [
            [0, 0, 160, 160],
            [0, 0, 320, 320],
            [0, 0, 160, 160],
            [120, 200, 760, 840],  # should be size 640x640
            [0, 0, 640, 640],  # should be size 640x640, but in corner
            [440, 1280, 1080, 1920],  # other corner
        ]
    )
    expected_crop_factor = np.array(
        [
            1,
            2,
            1,
            4,
            4,
            4,
        ]
    )  # 640 / 160 == 4

    for roi_, exp_bbox_, exp_crop_factor_ in zip(
        roi, expected_bbox, expected_crop_factor
    ):
        roi_ = roi_[None, :]
        exp_bbox_ = exp_bbox_[None, :]
        exp_crop_factor_ = exp_crop_factor_[None]
        bbox, crop_factor = PVN3D_E2E.get_crop_index(
            roi_, img_h, img_w, resnet_input_shape[0], resnet_input_shape[1]
        )
        assert_array_equal(
            bbox,
            exp_bbox_,
            err_msg=f"bbox size: ({bbox[:,2]-bbox[:,0]}, {bbox[:,3]-bbox[:,1]})",
        )
        assert_array_equal(crop_factor, exp_crop_factor_)

    # @staticmethod
    # def transform_indices_from_full_image_cropped(
    #     sampled_inds_in_original_image, bbox, crop_factor
    # ):
    #     crop_top_left = tf.concat(
    #         (tf.zeros((b, 1), tf.int32), bbox[:, :2]), -1
    #     )  # [b, 3]
    #     sampled_inds_in_roi = (
    #         sampled_inds_in_original_image - crop_top_left[:, tf.newaxis, :]
    #     )  #  [b, num_points, 3]
    #     sampled_inds_in_roi = (
    #         sampled_inds_in_roi / crop_factor[:, tf.newaxis, tf.newaxis]
    #     )  # [b, num_points, 3]
    #     sampled_inds_in_roi = tf.cast(sampled_inds_in_roi, tf.int32)
    #     return sampled_inds_in_roi


def test_index_transform():
    img_h = 1080
    img_w = 1920
    # theses boxes are from crop_index, so they are in the same aspect ratio with integer crop factor
    bbox = np.array(
        [
            [0, 0, 160, 160],
            [0, 0, 320, 320],
        ]
    )
    crop_factor = np.array(
        [
            1,
            2,
        ]
    )

    #  [b, num_points, 3], last dimensions contains inds in [batch, h, w]
    sampled_inds_in_original_image = np.array(
        [  # batch
            [  # list of points
                [0, 0, 0],  # top left corner
                [0, 79, 79],
                [0, 159, 159],  # bottom right corner
            ],
            [
                [0, 0, 0],  # top left corner
                [0, 159, 159],  # center point
                [0, 319, 319],  # bottom right corner
            ],
        ]
    )

    sampled_inds_in_roi = PVN3D_E2E.transform_indices_from_full_image_cropped(
        sampled_inds_in_original_image, bbox, crop_factor
    )

    expected_sampled_inds_in_roi = np.array(
        [  # batch
            [  # list of sampled indices
                [0, 0, 0],
                [0, 79, 79],
                [0, 159, 159],
            ],
            [
                [0, 0, 0],
                [0, 79, 79],  # center point
                [0, 159, 159],  # still bottom right corner of smaller bounding box
            ],
        ]
    )

    assert_array_equal(sampled_inds_in_roi, expected_sampled_inds_in_roi)
