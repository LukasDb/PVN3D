from typing import Dict, List
import tensorflow as tf
import numpy as np
from .pvn3d import PVN3D as _PVN3D
from .pprocessnet import _InitialPoseModel
from .utils import match_choose_adp, match_choose


class PVN3D_E2E(_PVN3D):
    """PVN3D with pre-processing layers.
    Can be called directly using a full RGB-D image, intrinsic matrix, ROI (y1,x1, y2, x2) and the mesh_kpts as input
    """

    def __init__(
        self,
        *,
        resnet_input_shape: List[int],
        num_kpts: int,
        num_cls: int,
        num_cpts: int,
        dim_xyz: int,
        resnet_params: Dict,
        psp_params: Dict,
        point_net2_params: Dict,
        dense_fusion_params: Dict,
        mlp_params: Dict,
        n_point_candidates: int,
    ):
        super().__init__(
            resnet_input_shape=resnet_input_shape,
            num_kpts=num_kpts,
            num_cls=num_cls,
            num_cpts=num_cpts,
            dim_xyz=dim_xyz,
            resnet_params=resnet_params,
            psp_params=psp_params,
            point_net2_params=point_net2_params,
            dense_fusion_params=dense_fusion_params,
            mlp_params=mlp_params,
            build=False,
        )
        self.resnet_input_shape = tf.cast(resnet_input_shape, tf.int32)

        self.initial_pose_model = _InitialPoseModel(n_point_candidates)

        b = 5
        mock_rgb = tf.random.uniform((b, 1080, 1920, 3), maxval=255, dtype=tf.int32)
        mock_dpt = tf.random.uniform((b, 1080, 1920, 1), maxval=5.0)
        mock_cam_intrinsic = tf.constant(
            [
                [
                    [572.411376953125, 0.0, 320.0],
                    [0.0, 573.5704345703125, 240.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
            * b,
            dtype=tf.float32,
        )

        # mock_roi = tf.random.uniform((b, 4), maxval=480, dtype=tf.int32)
        mock_roi = tf.constant([[0, 0, 480, 640]] * b, dtype=tf.int32)  # y1, x1, y2, x2
        mock_mesh_kpts = tf.random.uniform((b, 9, 3), maxval=0.01)
        self(
            (mock_rgb, mock_dpt, mock_cam_intrinsic, mock_roi, mock_mesh_kpts),
            training=True,
        )
        self.summary()

    @staticmethod
    def compute_normal_map(depth, camera_matrix):
        kernel = tf.constant([[[[0.5, 0.5]], [[-0.5, 0.5]]], [[[0.5, -0.5]], [[-0.5, -0.5]]]])

        diff = tf.nn.conv2d(depth, kernel, 1, "SAME")

        fx, fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]  # [b,]
        f = tf.stack([fx, fy], axis=-1)[:, tf.newaxis, tf.newaxis, :]  # [b, 1, 1, 2]

        diff = diff * f / depth

        mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

        diff = tf.where(mask, diff, 0.0)

        # smooth = tf.constant(4)
        # kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
        # kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
        # kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
        # diff2 = tf.nn.conv2d(diff, kernel2, 1, "SAME")

        # mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32), kernel2, 1, "SAME")

        # diff2 = diff2 / mask_conv
        diff2 = diff

        ones = tf.ones(tf.shape(diff2)[:3])[..., tf.newaxis]
        v_norm = tf.concat([diff2, ones], axis=-1)

        v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
        v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

        # v_norm = -tf.image.resize_with_crop_or_pad(
        #    v_norm, tf.shape(depth)[1], tf.shape(depth)[2]
        # )  # pad and flip (towards cam)
        return -v_norm

    @staticmethod
    def pcld_processor_tf(rgb, depth, camera_matrix, roi, num_sample_points, depth_trunc=2.0):
        """This function calculats a pointcloud from a RGB-D image and returns num_sample_points
        points from the pointcloud, randomly selected in the ROI specified.

        Args:
            rgb (b,h,w,3): RGB image with values in [0,255]
            depth (b,h,w,1): Depth image with float values in meters
            camera_matrix (b,3,3): Intrinsic camera matrix in OpenCV format
            roi (b,4): Region of Interest in the image (y1,x1,y2,x2)
            num_sample_points (int): Number of sampled points per image
            depth_trunc (float, optional): Truncate the depth image. Defaults to 2.0.

        Returns:
            xyz (b, num_sample_points, 3): Sampled pointcloud in m.
            feats (b, num_sample_points, 6): Feature pointcloud (RGB + normals)
            inds (b, num_sample_points, 3): Indices of the sampled points in the image
        """

        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        # normals = PVN3D_E2E.compute_normals(depth, camera_matrix) # [b, h*w, 3]
        normal_map = PVN3D_E2E.compute_normal_map(depth, camera_matrix)  # [b, h,w,3]

        h_depth = tf.shape(depth)[1]
        w_depth = tf.shape(depth)[2]
        x_map, y_map = tf.meshgrid(
            tf.range(w_depth, dtype=tf.int32), tf.range(h_depth, dtype=tf.int32)
        )
        y1 = y1[:, tf.newaxis, tf.newaxis]  # add, h, w dims
        x1 = x1[:, tf.newaxis, tf.newaxis]
        y2 = y2[:, tf.newaxis, tf.newaxis]
        x2 = x2[:, tf.newaxis, tf.newaxis]

        # invalidate outside of roi
        in_y = tf.logical_and(
            y_map[tf.newaxis, :, :] >= y1,
            y_map[tf.newaxis, :, :] < y2,
        )
        in_x = tf.logical_and(
            x_map[tf.newaxis, :, :] >= x1,
            x_map[tf.newaxis, :, :] < x2,
        )
        in_roi = tf.logical_and(in_y, in_x)

        # get masked indices (valid truncated depth inside of roi)
        is_valid = tf.logical_and(depth[..., 0] > 1e-6, depth[..., 0] < depth_trunc)
        inds = tf.where(tf.logical_and(in_roi, is_valid))
        inds = tf.cast(inds, tf.int32)  # [None, 3]

        inds = tf.random.shuffle(inds)

        # split index list into [b, None, 3] ragged tensor by using the batch index
        inds = tf.ragged.stack_dynamic_partitions(
            inds, inds[:, 0], tf.shape(rgb)[0]
        )  # [b, None, 3]

        # TODO if we dont have enough points, we pad the indices with 0s, how to handle that?
        inds = inds[:, :num_sample_points].to_tensor()  # [b, num_points, 3]

        # calculate xyz
        cam_cx, cam_cy = camera_matrix[:, 0, 2], camera_matrix[:, 1, 2]
        cam_fx, cam_fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]

        # inds[..., 0] == index into batch
        # inds[..., 1:] == index into y_map and x_map,  b times
        sampled_ymap = tf.gather_nd(y_map, inds[:, :, 1:])  # [b, num_points]
        sampled_xmap = tf.gather_nd(x_map, inds[:, :, 1:])  # [b, num_points]
        sampled_ymap = tf.cast(sampled_ymap, tf.float32)
        sampled_xmap = tf.cast(sampled_xmap, tf.float32)

        # z = tf.gather_nd(roi_depth, inds)  # [b, num_points]
        z = tf.gather_nd(depth[..., 0], inds)  # [b, num_points]
        x = (sampled_xmap - cam_cx[:, tf.newaxis]) * z / cam_fx[:, tf.newaxis]
        y = (sampled_ymap - cam_cy[:, tf.newaxis]) * z / cam_fy[:, tf.newaxis]
        xyz = tf.stack((x, y, z), axis=-1)

        rgb_feats = tf.gather_nd(rgb, inds)
        normal_feats = tf.gather_nd(normal_map, inds)
        feats = tf.concat([rgb_feats, normal_feats], -1)

        return xyz, feats, inds

    @staticmethod
    def get_crop_index(roi, in_h, in_w, resnet_h, resnet_w):
        """Given a ROI [y1,x1,y2,x2] in an image with dimensions [in_h, in_w]]
        this function returns the indices and the integer crop factor to crop the image
        according to the original roi, but with the same aspect ratio as [resnet_h, resnet_w]
        and scaled to integer multiple of [resnet_h,resnet_w].
        The resulting crop is centered around the roi center and encompases the whole roi.
        Additionally, the crop indices do not exceed the image dimensions.

        Args:
            roi (b,4): Region of Interest in the image (y1,x1,y2,x2)
            in_h (b,): Batchwise image height
            in_w (b,): Batchwise image width
            resnet_h (int): Height of the resnet input
            resnet_w (int): Width of the resnet input

        Returns:
            bbox (b, 4): Modified bounding boxes
            crop_factor (b,): Integer crop factor
        """

        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        x_c = tf.cast((x1 + x2) / 2, tf.int32)
        y_c = tf.cast((y1 + y2) / 2, tf.int32)

        bbox_w, bbox_h = (x2 - x1), (y2 - y1)
        w_factor = bbox_w / resnet_w  # factor to scale down to resnet shape
        h_factor = bbox_h / resnet_h
        crop_factor = tf.cast(tf.math.ceil(tf.maximum(w_factor, h_factor)), tf.int32)

        crop_w = resnet_w * crop_factor
        crop_h = resnet_h * crop_factor

        x1_new = x_c - tf.cast(crop_w / 2, tf.int32)
        x2_new = x_c + tf.cast(crop_w / 2, tf.int32)
        y1_new = y_c - tf.cast(crop_h / 2, tf.int32)
        y2_new = y_c + tf.cast(crop_h / 2, tf.int32)

        x2_new = tf.where(x1_new < 0, crop_w, x2_new)
        x1_new = tf.where(x1_new < 0, 0, x1_new)

        x1_new = tf.where(x2_new > in_w, in_w - crop_w, x1_new)
        x2_new = tf.where(x2_new > in_w, in_w, x2_new)

        y2_new = tf.where(y1_new < 0, crop_h, y2_new)
        y1_new = tf.where(y1_new < 0, 0, y1_new)

        y1_new = tf.where(y2_new > in_h, in_h - crop_h, y1_new)
        y2_new = tf.where(y2_new > in_h, in_h, y2_new)

        return tf.stack([y1_new, x1_new, y2_new, x2_new], axis=-1), crop_factor

    @staticmethod
    def transform_indices_from_full_image_cropped(
        sampled_inds_in_original_image, bbox, crop_factor
    ):
        """Transforms indices from full image to croppend and rescaled images.
        Original indices [b, h, w, 3] with the last dimensions as indices into [b, h, w]
        are transformed and have same shape [b,h,w,3], however the indices now index
        into the cropped and rescaled images according to the bbox and crop_factor

        To be used with tf.gather_nd
        Since the first index refers to the batch, no batch_dims is needed

        Examples:
            Index: [500,500] with bounding box [500,500,...] is transform to [0,0]
            index [2, 2] with bounding box [0, 0] and crop_factor 2 is transformed to [1,1]


        Args:
            sampled_inds_in_original_image (b,h,w,3): Indices into the original image
            bbox (b,4): Region of Interest in the image (y1,x1,y2,x2)
            crop_factor (b,): Integer crop factor

        Returns:
            sampled_inds_in_roi (b,h,w,3): Indices into the cropped and rescaled image
        """
        b = tf.shape(sampled_inds_in_original_image)[0]

        # sampled_inds_in_original_image: [b, num_points, 3]
        # with last dimension is index into [b, h, w]
        # crop_factor: [b, ]

        crop_top_left = tf.concat((tf.zeros((b, 1), tf.int32), bbox[:, :2]), -1)  # [b, 3]

        sampled_inds_in_roi = sampled_inds_in_original_image - crop_top_left[:, tf.newaxis, :]

        # apply scaling to indices, BUT ONLY H AND W INDICES (and not batch index)
        crop_factor_bhw = tf.concat(
            (
                tf.ones((b, 1), dtype=tf.int32),
                crop_factor[:, tf.newaxis],
                crop_factor[:, tf.newaxis],
            ),
            -1,
        )  # [b, 3]
        sampled_inds_in_roi = sampled_inds_in_roi / crop_factor_bhw[:, tf.newaxis, :]

        return tf.cast(sampled_inds_in_roi, tf.int32)

    @tf.function
    def call(self, inputs, training=None):
        (
            full_rgb,
            depth,
            intrinsics,
            roi,
            mesh_kpts,  # dont use keypoints from dataset
        ) = inputs  # rgb [b,h,w,3], depth: [b,h,w,1], intrinsics: [b, 3,3], roi: [b,4]
        h, w = tf.shape(full_rgb)[1], tf.shape(full_rgb)[2]

        # crop the image to the aspect ratio for resnet and integer crop factor
        bbox, crop_factor = self.get_crop_index(
            roi, h, w, self.resnet_input_shape[0], self.resnet_input_shape[1]
        )  # bbox: [b, 4], crop_factor: [b]

        xyz, feats, sampled_inds_in_original_image = self.pcld_processor_tf(
            tf.cast(full_rgb, tf.float32) / 255.0,
            depth,
            intrinsics,
            bbox,
            self.point_net2_params.n_sample_points,
        )

        sampled_inds_in_roi = self.transform_indices_from_full_image_cropped(
            sampled_inds_in_original_image, bbox, crop_factor
        )

        norm_bbox = tf.cast(bbox / [h, w, h, w], tf.float32)  # normalize bounding box
        cropped_rgbs = tf.image.crop_and_resize(
            tf.cast(full_rgb, tf.float32),
            norm_bbox,
            tf.range(tf.shape(full_rgb)[0]),
            self.resnet_input_shape[:2],
        )

        # stop gradients for preprocessing
        cropped_rgbs = tf.stop_gradient(cropped_rgbs)
        xyz = tf.stop_gradient(xyz)
        feats = tf.stop_gradient(feats)
        sampled_inds_in_roi = tf.stop_gradient(sampled_inds_in_roi)

        pcld_emb = self.pointnet2_model((xyz, feats), training=training)
        resnet_feats = self.resnet_model(cropped_rgbs, training=training)
        rgb_features = self.psp_model(resnet_feats, training=training)  # [b, h_res, w_res, c]
        rgb_emb = tf.gather_nd(rgb_features, sampled_inds_in_roi)
        feats_fused = self.dense_fusion_model([rgb_emb, pcld_emb], training=training)
        kp, seg, cp = self.mlp_model(feats_fused, training=training)

        if training:
            return kp, seg, cp, xyz, sampled_inds_in_original_image, mesh_kpts, cropped_rgbs
        else:
            batch_R, batch_t, voted_kpts = self.initial_pose_model([xyz, kp, cp, seg, mesh_kpts])
            return (
                batch_R,
                batch_t,
                voted_kpts,
                (kp, seg, cp, xyz, sampled_inds_in_original_image, mesh_kpts, cropped_rgbs),
            )
