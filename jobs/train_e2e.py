from tqdm import tqdm
from millify import millify
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d
import random

import cvde
from models.pvn3d_e2e import PVN3D_E2E
from datasets.blender import TrainBlender, ValBlender
from losses.pvn_loss import PvnLoss


class TrainE2E(cvde.job.Job):
    def run(self):
        job_cfg = self.config

        self.num_validate = job_cfg["num_validate"]
        
        strategy = tf.distribute.get_strategy()

        print("ACTIVE GPUS: ", tf.config.experimental.list_physical_devices("GPU"))
        print("Running job: ", self.name)

        with strategy.scope():
            # model: tf.keras.Model = load_model(job_cfg["Model"], **job_cfg)
            self.model = model = PVN3D_E2E(**job_cfg["PVN3D_E2E"])
            train_set = TrainBlender(**job_cfg["TrainBlender"])
            val_set = ValBlender(**job_cfg["ValBlender"])
        self.demo_set = ValBlender(**job_cfg["ValBlender"]).to_tf_dataset().take(self.num_validate)

        self.color_seg = val_set.color_seg
        self.mesh_vertices = np.asarray(val_set.mesh.sample_points_poisson_disk(1000).points)

        self.loss_fn = loss_fn = PvnLoss(**job_cfg["PvnLoss"])
        optimizer = tf.keras.optimizers.Adam(**job_cfg["Adam"])

        self.log_visualization(-1)

        num_epochs = job_cfg["epochs"]
        for epoch in range(num_epochs):
            bar = tqdm(total=len(train_set), desc=f"Train ({epoch}/{num_epochs})")

            loss_vals = {}
            for x, y in train_set.to_tf_dataset():
                if self.is_stopped:
                    return

                with tf.GradientTape() as tape:
                    pred = model(x, training=True)
                    loss_combined, loss_cp, loss_kp, loss_seg = loss_fn(y, pred)
                gradients = tape.gradient(loss_combined, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                loss_vals["loss"] = loss_vals.get("loss", []) + [loss_combined.numpy()]
                loss_vals["loss_cp"] = loss_vals.get("loss_cp", []) + [loss_cp.numpy()]
                loss_vals["loss_kp"] = loss_vals.get("loss_kp", []) + [loss_kp.numpy()]
                loss_vals["loss_seg"] = loss_vals.get("loss_seg", []) + [loss_seg.numpy()]

                bar.update(1)  # update by batchsize
                bar.set_postfix(
                    {
                        "loss": millify(loss_combined.numpy(), precision=4),
                    }
                )
            bar.close()
            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"train_{k}", loss_vals[k], epoch)

            loss_vals = {}
            for x, y in tqdm(
                val_set.to_tf_dataset(),
                desc=f"Val ({epoch}/{num_epochs})",
                total=len(val_set),
            ):
                _, _, _, pred = model(x, training=False)
                l = loss_fn(y, pred)
                for k, v in zip(["loss", "loss_cp", "loss_kp", "loss_seg"], l):
                    loss_vals[k] = loss_vals.get(k, []) + [v]

            for k, v in loss_vals.items():
                loss_vals[k] = tf.reduce_mean(v)
                self.tracker.log(f"val_{k}", loss_vals[k], epoch)

            self.log_visualization(epoch)

    def log_visualization(self, epoch: int, logs=None):
        i = 0
        for x, y in self.demo_set:
            (
                b_rgb,
                b_depth,
                b_intrinsics,
                b_roi,
                b_mesh_kpts,
            ) = x

            b_RT_gt, b_mask = y

            (
                b_R,
                b_t,
                b_kpts_pred,
                (b_kp_offset, b_seg_pred, b_cp_offset, b_xyz, b_sampled_inds, _, b_cropped_rgb),
            ) = self.model(x, training=False)

            h, w = tf.shape(b_rgb)[1], tf.shape(b_rgb)[2]
            b_roi, b_crop_factor = self.model.get_crop_index(
                b_roi,
                h,
                w,
                self.model.resnet_input_shape[0],
                self.model.resnet_input_shape[1],
            )

            b_rgb = b_rgb.numpy()
            b_roi = b_roi.numpy()
            b_mask = b_mask.numpy()
            b_seg_pred = b_seg_pred.numpy()
            b_sampled_inds = b_sampled_inds.numpy()
            b_kp_offset = b_kp_offset.numpy()
            b_cropped_rgb = b_cropped_rgb.numpy()
            b_cp_offset = b_cp_offset.numpy()
            b_offsets = np.concatenate([b_kp_offset, b_cp_offset], axis=2)

            b_kpts_gt = (
                tf.matmul(b_mesh_kpts, tf.transpose(b_RT_gt[:, :3, :3], (0, 2, 1)))
                + b_RT_gt[:, tf.newaxis, :3, 3]
            )
            b_kpts_gt = self.project_batch_to_image(b_kpts_gt, b_intrinsics)
            b_kpts_pred = self.project_batch_to_image(b_kpts_pred, b_intrinsics)

            b_R = b_R.numpy()
            b_t = b_t.numpy()
            b_RT_gt = b_RT_gt.numpy()
            b_intrinsics = b_intrinsics.numpy()
            b_crop_factor = b_crop_factor.numpy()

            # draw mesh
            # create a batch of mesh_vertices
            b_mesh_vertices = np.tile(self.mesh_vertices, (b_R.shape[0], 1, 1))
            # transform mesh_vertices
            b_mesh_vertices = (
                np.matmul(b_mesh_vertices, b_R[:, :3, :3].transpose((0, 2, 1)))
                + b_t[:, tf.newaxis, :]
            )
            b_mesh_vertices = self.project_batch_to_image(b_mesh_vertices, b_intrinsics)[
                ..., :2
            ].astype(np.int32)

            # get gt offsets
            b_mask_selected = tf.gather_nd(b_mask, b_sampled_inds)
            b_kp_offsets_gt, b_cp_offsets_gt = self.loss_fn.get_offst(
                b_RT_gt,
                b_xyz,
                b_mask_selected,
                b_mesh_kpts,
            )
            b_offsets_gt = np.concatenate([b_kp_offsets_gt, b_cp_offsets_gt], axis=2)

            for (
                rgb,
                crop_rgb,
                roi,
                seg_pred,
                sampled_inds,
                kpts_gt,
                kpts_pred,
                mesh_vertices,
                offsets,
                offsets_gt,
                crop_factor,
            ) in zip(
                b_rgb,
                b_cropped_rgb,
                b_roi,
                b_seg_pred,
                b_sampled_inds,
                b_kpts_gt,
                b_kpts_pred,
                b_mesh_vertices,
                b_offsets,
                b_offsets_gt,
                b_crop_factor,
            ):
                vis_seg = self.draw_segmentation(rgb.copy(), sampled_inds, seg_pred, roi)
                self.tracker.log(f"RGB ({i})", vis_seg, index=epoch)
                
                self.tracker.log(f"RGB (crop) ({i})", crop_rgb.astype(np.uint8), index=epoch)

                vis_mesh = self.draw_object_mesh(rgb.copy(), roi, mesh_vertices)
                self.tracker.log(f"RGB (mesh) ({i})", vis_mesh, index=epoch)

                vis_kpts = self.draw_keypoint_correspondences(rgb.copy(), roi, kpts_gt, kpts_pred)
                self.tracker.log(f"RGB (kpts) ({i})", vis_kpts, index=epoch)

                vis_offsets = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets,
                    sampled_inds,
                    kpts_gt,
                    radius=int(crop_factor),
                )
                vis_offsets_gt = self.draw_keypoint_offsets(
                    rgb.copy(),
                    roi,
                    offsets_gt,
                    sampled_inds,
                    kpts_gt,
                    radius=int(crop_factor),
                )
                # assemble images side-by-side
                vis_offsets = np.concatenate([vis_offsets, vis_offsets_gt], axis=1)
                self.tracker.log(f"RGB (offsets) ({i})", vis_offsets, index=epoch)

                i = i + 1
                if i >= self.num_validate:
                    return

    def crop_to_roi(self, rgb, roi, margin=50):
        (
            y1,
            x1,
            y2,
            x2,
        ) = roi[:4]
        h, w = rgb.shape[:2]
        x1 = np.clip(x1 - margin, 0, w)
        x2 = np.clip(x2 + margin, 0, w)
        y1 = np.clip(y1 - margin, 0, h)
        y2 = np.clip(y2 + margin, 0, h)
        return rgb[y1:y2, x1:x2]

    def draw_segmentation(self, rgb, sampled_inds, seg_pred, roi):
        (
            y1,
            x1,
            y2,
            x2,
        ) = roi[:4]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw gray cirlces at sampled inds
        # sampled_inds: [num_points, 3]
        for (_, h_ind, w_ind), seg in zip(sampled_inds, seg_pred):
            color = (0, 0, 255) if seg > 0.5 else (255, 0, 0)
            cv2.circle(rgb, (w_ind, h_ind), 1, color, -1)
        return rgb

    def draw_object_mesh(self, rgb, roi, mesh_vertices):
        h, w = rgb.shape[:2]
        clipped_mesh_vertices = np.clip(mesh_vertices, 0, [w - 1, h - 1])
        for x, y in clipped_mesh_vertices:
            cv2.circle(rgb, (x, y), 1, (0, 0, 255), -1)
        return self.crop_to_roi(rgb, roi)

    def draw_keypoint_correspondences(self, rgb, roi, kpts_gt, kpts_pred):
        # normalize z_coords of keypoints to [0, 1]
        kpts_gt[..., 2] = (kpts_gt[..., 2] - np.min(kpts_gt[..., 2])) / (
            np.max(kpts_gt[..., 2]) - np.min(kpts_gt[..., 2])
        )
        kpts_pred[..., 2] = (kpts_pred[..., 2] - np.min(kpts_pred[..., 2])) / (
            np.max(kpts_pred[..., 2]) - np.min(kpts_pred[..., 2])
        )

        for (x_gt, y_gt, z_gt), (x_pred, y_pred, z_pred) in zip(kpts_gt, kpts_pred):
            gt_color = np.array((0, 255, 0), dtype=np.uint8)
            pred_color = np.array((255, 0, 0), dtype=np.uint8)

            scale_marker = lambda z: 10 + int(z * 20)

            cv2.drawMarker(
                rgb,
                (int(x_gt), int(y_gt)),
                gt_color.tolist(),
                cv2.MARKER_CROSS,
                scale_marker(z_gt),
                1,
            )
            cv2.drawMarker(
                rgb,
                (int(x_pred), int(y_pred)),
                pred_color.tolist(),
                cv2.MARKER_TILTED_CROSS,
                scale_marker(z_pred),
                1,
            )
            cv2.line(
                rgb,
                (int(x_gt), int(y_gt)),
                (int(x_pred), int(y_pred)),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return self.crop_to_roi(rgb, roi)

    def project_batch_to_image(self, pts, b_intrinsics):
        cam_cx, cam_cy = b_intrinsics[:, 0, 2], b_intrinsics[:, 1, 2]  # [b]
        cam_fx, cam_fy = b_intrinsics[:, 0, 0], b_intrinsics[:, 1, 1]  # [b]
        coors = (
            pts[..., :2] / pts[..., 2:] * tf.stack([cam_fx, cam_fy], axis=1)[:, tf.newaxis, :]
            + tf.stack([cam_cx, cam_cy], axis=1)[:, tf.newaxis, :]
        )
        coors = tf.floor(coors)
        return tf.concat([coors, pts[..., 2:]], axis=-1).numpy()

    def draw_keypoint_offsets(self, rgb, roi, offsets, sampled_inds, kpts_gt, radius=1):
        cropped_rgb = self.crop_to_roi(rgb, roi, margin=10)
        h, w = cropped_rgb.shape[:2]
        vis_offsets = np.zeros((h * 3, w * 3, 3), dtype=np.uint8).astype(np.uint8)

        for i in range(9):
            offset_view = np.zeros_like(rgb, dtype=np.uint8)

            # get color hue from offset
            hue = np.arctan2(offsets[:, i, 1], offsets[:, i, 0]) / np.pi
            hue = (hue + 1) / 2
            hue = (hue * 180).astype(np.uint8)
            # value = np.ones_like(hue) * 255
            value = (np.linalg.norm(offsets[:, i, :], axis=-1) / 0.1 * 255).astype(np.uint8)
            hsv = np.stack([hue, np.ones_like(hue) * 255, value], axis=-1)
            colors_offset = cv2.cvtColor(hsv[None], cv2.COLOR_HSV2RGB).astype(np.uint8)[0]

            for (_, h_ind, w_ind), color in zip(sampled_inds, colors_offset):
                cv2.circle(
                    offset_view,
                    (w_ind, h_ind),
                    radius,
                    tuple(map(int, color)),
                    -1,
                )

            # # mark correct keypoint
            cv2.drawMarker(
                offset_view,
                (int(kpts_gt[i, 0]), int(kpts_gt[i, 1])),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=1,
            )

            offset_view = self.crop_to_roi(offset_view, roi, margin=10)
            vis_offsets[
                h * (i // 3) : h * (i // 3 + 1), w * (i % 3) : w * (i % 3 + 1)
            ] = offset_view
        return vis_offsets