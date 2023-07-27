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

        diff = tf.nn.conv2d(depth, kernel, 1, "VALID")

        fx, fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]
        scale_depth = tf.concat(
            [
                depth / tf.reshape(fx, [-1, 1, 1, 1]),
                depth / tf.reshape(fy, [-1, 1, 1, 1]),
            ],
            -1,
        )

        # clip=tf.constant(1)
        # diff = tf.clip_by_value(diff, -clip, clip)
        diff = diff / scale_depth[:, :-1, :-1, :]  # allow nan -> filter later

        mask = tf.logical_and(~tf.math.is_nan(diff), tf.abs(diff) < 5)

        diff = tf.where(mask, diff, 0.0)

        smooth = tf.constant(4)
        kernel2 = tf.cast(tf.tile([[1 / tf.pow(smooth, 2)]], (smooth, smooth)), tf.float32)
        kernel2 = tf.expand_dims(tf.expand_dims(kernel2, axis=-1), axis=-1)
        kernel2 = kernel2 * tf.eye(2, batch_shape=(1, 1))
        diff2 = tf.nn.conv2d(diff, kernel2, 1, "VALID")

        mask_conv = tf.nn.conv2d(tf.cast(mask, tf.float32), kernel2, 1, "VALID")

        diff2 = diff2 / mask_conv

        ones = tf.expand_dims(tf.ones(tf.shape(diff2)[:3]), -1)
        v_norm = tf.concat([diff2, ones], axis=-1)

        v_norm, _ = tf.linalg.normalize(v_norm, axis=-1)
        v_norm = tf.where(~tf.math.is_nan(v_norm), v_norm, [0])

        v_norm = -tf.image.resize_with_crop_or_pad(
            v_norm, tf.shape(depth)[1], tf.shape(depth)[2]
        )  # pad and flip (towards cam)
        return v_norm

    @staticmethod
    def pcld_processor_tf(rgb, depth, camera_matrix, roi, num_sample_points, depth_trunc=2.0):
        # depth: [b, h, w, 1]
        # rgb: [b, h, w, 3]
        # camera_matrix: [b, 3, 3]
        # roi: [b, 4] (y1, x1, y2, x2)

        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        # normals = PVN3D_E2E.compute_normals(depth, camera_matrix) # [b, h*w, 3]
        normal_map = PVN3D_E2E.compute_normal_map(depth, camera_matrix)  # [b, h,w,3]

        h_depth = tf.shape(depth)[1]
        w_depth = tf.shape(depth)[2]
        x_map, y_map = tf.meshgrid(
            tf.range(w_depth, dtype=tf.int32), tf.range(h_depth, dtype=tf.int32)
        )

        # invalidate outside of roi
        in_y = tf.logical_and(
            y_map[tf.newaxis, :, :] >= y1[:, tf.newaxis, tf.newaxis],
            y_map[tf.newaxis, :, :] < y2[:, tf.newaxis, tf.newaxis],
        )
        in_x = tf.logical_and(
            x_map[tf.newaxis, :, :] >= x1[:, tf.newaxis, tf.newaxis],
            x_map[tf.newaxis, :, :] < x2[:, tf.newaxis, tf.newaxis],
        )
        in_roi = tf.logical_and(in_y, in_x)

        # get masked indices (valid truncated depth inside of roi)
        is_valid_depth = tf.logical_and(depth[..., 0] > 1e-6, depth[..., 0] < depth_trunc)
        inds = tf.where(
            tf.logical_and(in_roi, is_valid_depth)
        )  # [b*h*w, 3], last dimension is [batch_index, y_index, x_index]
        inds = tf.cast(inds, tf.int32)

        inds = tf.random.shuffle(inds)

        inds = tf.ragged.stack_dynamic_partitions(
            inds, inds[:, 0], tf.shape(rgb)[0]
        )  # [b, None, 3]

        # TODO if we dont have enough points, we pad the indices with 0s, how to handle that?
        inds = inds[:, :num_sample_points].to_tensor()  # [b, num_points, 3]

        # calculate xyz
        cam_cx, cam_cy = camera_matrix[:, 0, 2], camera_matrix[:, 1, 2]
        cam_fx, cam_fy = camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]

        # inds[..., 0] == index into batch -> not needed here
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
        y1, x1, y2, x2 = roi[:, 0], roi[:, 1], roi[:, 2], roi[:, 3]

        x_c = tf.cast((x1 + x2) / 2, tf.int32)
        y_c = tf.cast((y1 + y2) / 2, tf.int32)

        bbox_w, bbox_h = (x2 - x1), (y2 - y1)
        w_factor = bbox_w / resnet_w  # factor to scale down to resnet shape
        h_factor = bbox_h / resnet_h
        crop_factor = tf.cast(tf.math.ceil(tf.maximum(w_factor, h_factor)), tf.int32)

        resnet_w = resnet_w * crop_factor
        resnet_h = resnet_h * crop_factor

        x1_new = x_c - tf.cast(resnet_w / 2, tf.int32)
        x2_new = x_c + tf.cast(resnet_w / 2, tf.int32)
        y1_new = y_c - tf.cast(resnet_h / 2, tf.int32)
        y2_new = y_c + tf.cast(resnet_h / 2, tf.int32)

        x2_new = tf.where(x1_new < 0, resnet_w, x2_new)
        x1_new = tf.where(x1_new < 0, 0, x1_new)

        x1_new = tf.where(x2_new > in_w, in_w - resnet_w, x1_new)
        x2_new = tf.where(x2_new > in_w, in_w, x2_new)

        y2_new = tf.where(y1_new < 0, resnet_h, y2_new)
        y1_new = tf.where(y1_new < 0, 0, y1_new)

        y1_new = tf.where(y2_new > in_h, in_h - resnet_h, y1_new)
        y2_new = tf.where(y2_new > in_h, in_h, y2_new)

        return tf.stack([y1_new, x1_new, y2_new, x2_new], axis=-1), crop_factor

    @staticmethod
    def transform_indices_from_full_image_cropped(
        sampled_inds_in_original_image, bbox, crop_factor
    ):
        b = tf.shape(sampled_inds_in_original_image)[0]

        # sampled_inds_in_original_image: [b, num_points, 3]
        # with last dimension is index into [b, h, w]
        # crop_factor: [b, ]

        crop_top_left = tf.concat((tf.zeros((b, 1), tf.int32), bbox[:, :2]), -1)  # [b, 3]

        sampled_inds_in_roi = sampled_inds_in_original_image - crop_top_left[:, tf.newaxis, :]

        # apply scaling to indices, BUT ONLY H AND W
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
            mesh_kpts,
        ) = inputs  # rgb [b,h,w,3], depth: [b,h,w,1], intrinsics: [b, 3,3], roi: [b,4]
        h, w = tf.shape(full_rgb)[1], tf.shape(full_rgb)[2]

        if training:
            # inject noise to the roi
            roi = roi + tf.random.uniform(tf.shape(roi), -20, 20, dtype=tf.int32)

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
            full_rgb,
            norm_bbox,
            tf.range(tf.shape(full_rgb)[0]),
            self.resnet_input_shape[:2],
        )

        pcld_emb = self.pointnet2_model((xyz, feats), training=training)
        resnet_feats = self.resnet_model(cropped_rgbs, training=training)
        rgb_features = self.psp_model(resnet_feats, training=training)  # [b, h_res, w_res, c]
        rgb_emb = tf.gather_nd(rgb_features, sampled_inds_in_roi)
        feats_fused = self.dense_fusion_model([rgb_emb, pcld_emb], training=training)
        kp, seg, cp = self.mlp_model(feats_fused, training=training)

        if not training:
            batch_R, batch_t, voted_kpts = self.initial_pose_model([xyz, kp, cp, seg, mesh_kpts])
            return (
                batch_R,
                batch_t,
                voted_kpts,
                (kp, seg, cp, xyz, sampled_inds_in_original_image, mesh_kpts, cropped_rgbs),
            )

        return kp, seg, cp, xyz, sampled_inds_in_original_image, mesh_kpts, cropped_rgbs
