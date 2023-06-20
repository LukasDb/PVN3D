import numpy as np
import cv2
import tensorflow as tf
from cvde.job.job_tracker import JobTracker
from cvde.tf import callback as cb
from datasets.blender import ValBlender


class DemoInference(cb.Callback):
    def __init__(self, tracker: JobTracker, **kwargs):
        super().__init__(tracker, **kwargs)
        self.num_validate = kwargs["num_validate"]
        ds = ValBlender(**kwargs["data_cfg"])
        self.demo_set = ds.to_tf_dataset().take(self.num_validate)
        self.color_seg = ds.color_seg

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1, logs=logs)

    def on_epoch_end(self, epoch: int, logs=None):
        i = 0
        for x, y in self.demo_set:
            (
                b_rgb,
                b_depth,
                b_intrinsics,
                b_roi,
                b_mesh_kpts,
            ) = x
            b_rt, b_mask = y

            (
                b_R,
                b_t,
                b_kpts_pred,
                (_, b_seg_pred, _, _, b_sampled_inds, _),
            ) = self.model(x, training=False)

            b_rgb = b_rgb.numpy()
            b_roi = b_roi.numpy()
            b_seg_pred = b_seg_pred.numpy()
            b_sampled_inds = b_sampled_inds.numpy()

            # visualize keypoints gt vs prediction in 3D ?
            # 3D not really possible -> draw gt and pred on image and connect with lines for correspondence
            # 1) transform mesh keypoints to image space
            # 2) draw keypoints on image
            # 3) draw lines between gt and pred keypoints

            # b_mesh_kpts: [9, 3] # in object frame
            # b_intrinsics: [3, 3]
            # b_rt: [4, 4]
            # b_kpts_pred: [b, 9, 3]

            b_kpts_gt = (
                tf.matmul(b_mesh_kpts, b_rt[:, :3, :3]) + b_rt[:, tf.newaxis, :3, 3]
            )  # [b, 9,3]
            cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]  # [b]
            cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]  # [b]
            b_kpts_gt = b_kpts_gt[:, :, :2] / b_kpts_gt[:, :, 2:]  # [b, 9, 2]
            b_kpts_gt = (
                b_kpts_gt * tf.stack([cam_fx, cam_fy], axis=1)[:, tf.newaxis, :]
                + tf.stack([cam_cx, cam_cy], axis=1)[:, tf.newaxis, :]
            )  # [b, 9, 2]
            b_kpts_gt = b_kpts_gt.numpy().astype(int)

            b_kpts_pred = b_kpts_pred[:, :, :2] / b_kpts_pred[:, :, 2:]  # [b, 9, 2]
            b_kpts_pred = (
                b_kpts_pred * tf.stack([cam_fx, cam_fy], axis=1)[:, tf.newaxis, :]
                + tf.stack([cam_cx, cam_cy], axis=1)[:, tf.newaxis, :]
            )  # [b, 9, 2]
            b_kpts_pred = b_kpts_pred.numpy().astype(int)

            b_R = b_R.numpy()
            b_t = b_t.numpy()
            b_rt = b_rt.numpy()
            b_intrinsics = b_intrinsics.numpy()

            for (
                rgb,
                roi,
                seg_pred,
                sampled_inds,
                kpts_gt,
                kpts_pred,
                rt_gt,
                intrinsics,
                r_pred,
                t_pred,
            ) in zip(
                b_rgb,
                b_roi,
                b_seg_pred,
                b_sampled_inds,
                b_kpts_gt,
                b_kpts_pred,
                b_rt,
                b_intrinsics,
                b_R,
                b_t,
            ):
                # seg pred :  [n_pts, 1] # binary segmentation
                (
                    y1,
                    x1,
                    y2,
                    x2,
                ) = roi[:4]
                rgb2 = rgb.copy()
                cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # draw gray cirlces at sampled inds
                # sampled_inds: [num_points, 3]
                for (_, h_ind, w_ind), seg in zip(sampled_inds, seg_pred):
                    color = (0, 0, 255) if seg > 0.5 else (255, 0, 0)
                    cv2.circle(rgb, (w_ind, h_ind), 1, color, -1)

                self.log_image(f"RGB ({i})", rgb)

                for (x1, y1), (x2, y2) in zip(kpts_gt, kpts_pred):
                    cv2.drawMarker(rgb2, (x1, y1), (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
                    cv2.drawMarker(rgb2, (x2, y2), (255, 0, 0), cv2.MARKER_CROSS, 25, 1)
                    cv2.line(rgb2, (x1, y1), (x2, y2), (255, 255, 255), 1)

        
                self.log_image(f"RGB (kpts) ({i})", rgb2)

                i = i + 1
                if i >= self.num_validate:
                    return
