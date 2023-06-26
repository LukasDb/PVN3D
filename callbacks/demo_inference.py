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

            cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]  # [b]
            cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]  # [b]

            def to_image(pts):
                coors = (
                    pts[..., :2]
                    / pts[..., 2:]
                    * tf.stack([cam_fx, cam_fy], axis=1)[:, tf.newaxis, :]
                    + tf.stack([cam_cx, cam_cy], axis=1)[:, tf.newaxis, :]
                )
                coors = tf.floor(coors)
                return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

            b_kpts_gt = (
                tf.matmul(b_mesh_kpts, tf.transpose(b_rt[:, :3, :3], (0, 2, 1)))
                + b_rt[:, tf.newaxis, :3, 3]
            )
            b_kpts_gt = to_image(b_kpts_gt)
            b_kpts_pred = to_image(b_kpts_pred)

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
                kpts_pred
            ) in zip(
                b_rgb,
                b_roi,
                b_seg_pred,
                b_sampled_inds,
                b_kpts_gt,
                b_kpts_pred
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

                # normalize z_coords of keypoints to [0, 1]
                kpts_gt[..., 2] = (kpts_gt[..., 2] - np.min(kpts_gt[..., 2])) / (
                    np.max(kpts_gt[..., 2]) - np.min(kpts_gt[..., 2])
                )
                kpts_pred[..., 2] = (kpts_pred[..., 2] - np.min(kpts_pred[..., 2])) / (
                    np.max(kpts_pred[..., 2]) - np.min(kpts_pred[..., 2])
                )

                for (x_gt, y_gt, z_gt), (x_pred, y_pred, z_pred) in zip(
                    kpts_gt, kpts_pred
                ):
                    gt_color = np.array((0, 255, 0), dtype=np.uint8)
                    pred_color = np.array((255, 0, 0), dtype=np.uint8)

                    scale_marker = lambda z: 10 + int(z * 20)

                    cv2.drawMarker(
                        rgb2,
                        (int(x_gt), int(y_gt)),
                        gt_color.tolist(),
                        cv2.MARKER_CROSS,
                        scale_marker(z_gt),
                        1,
                    )
                    cv2.drawMarker(
                        rgb2,
                        (int(x_pred), int(y_pred)),
                        pred_color.tolist(),
                        cv2.MARKER_TILTED_CROSS,
                        scale_marker(z_pred),
                        1,
                    )
                    cv2.line(
                        rgb2,
                        (int(x_gt), int(y_gt)),
                        (int(x_pred), int(y_pred)),
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                # crop to region of interest (with margin)
                h, w = rgb2.shape[:2]
                x1 = np.clip(x1 - 100, 0, w)
                x2 = np.clip(x2 + 100, 0, w)
                y1 = np.clip(y1 - 100, 0, h)
                y2 = np.clip(y2 + 100, 0, h)
                rgb2 = rgb2[y1:y2, x1:x2]

                self.log_image(f"RGB (kpts) ({i})", rgb2)

                i = i + 1
                if i >= self.num_validate:
                    return
