import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss as _BinaryFocalLoss


class PvnLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        *,
        binary_loss,
        kp_loss_discount,
        cp_loss_discount,
        seg_loss_discount,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.binary_loss = binary_loss
        self.kp_loss_discount = kp_loss_discount
        self.cp_loss_discount = cp_loss_discount
        self.seg_loss_discount = seg_loss_discount
        red = tf.keras.losses.Reduction.SUM
        self.BinaryFocalLoss = _BinaryFocalLoss(
            gamma=2, from_logits=True, reduction=red
        )
        self.CategoricalCrossentropy = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=red
        )

    @staticmethod
    def get_offst(
        RT,  # [b, 4,4]
        pcld_xyz,  # [b, n_pts, 3]
        mask_selected,  # [b, n_pts, 1] 0|1
        kpts_cpts,  # [b, 9,3]
    ):
        # transform kpts_cpts to camera frame using rt
        kpts_cpts_cam = (
            tf.matmul(kpts_cpts, tf.transpose(RT[:, :3, :3], (0, 2, 1)))
            + RT[:, tf.newaxis, :3, 3]
        )

        # calculate offsets to the pointcloud
        kpts_cpts_cam = tf.expand_dims(kpts_cpts_cam, axis=1)  # [b, 1, 9, 3]
        pcld_xyz = tf.expand_dims(pcld_xyz, axis=2)  # [b, n_pts, 1, 3]
        offsets = tf.subtract(kpts_cpts_cam, pcld_xyz)  # [b, n_pts, 9, 3]
        # mask offsets to the object points
        # offsets = offsets * tf.cast(mask_selected[:, :, tf.newaxis], tf.float32)
        offsets = tf.where(mask_selected[:, :, tf.newaxis] == 1, offsets, 0.0)
        kp_offsets = offsets[:, :, :8, :]  # [b, n_pts, 8, 3]
        cp_offsets = offsets[:, :, 8:, :]  # [b, n_pts, 1, 3]
        return kp_offsets, cp_offsets

    @staticmethod
    def l1_loss(offset_pred, offset_gt, mask_labels):
        """
        :param: pred_ofsts: [bs, n_pts, n_kpts, c] or [bs, n_pts, n_cpts, c]
                targ_ofst: [bs, n_pts, n_kpts, c] for kp,  [bs, n_pts, n_cpts, c] for cp
                mask_labels: [bs, n_pts]
        """
        bs, n_pts, n_kpts = (
            tf.shape(offset_pred)[0],
            tf.shape(offset_pred)[1],
            tf.shape(offset_pred)[2],
        )
        num_nonzero = tf.cast(tf.math.count_nonzero(mask_labels), tf.float32)

        w = tf.cast(mask_labels, dtype=tf.float32)
        w = tf.reshape(w, shape=[bs, n_pts, 1, 1])
        w = tf.repeat(w, repeats=n_kpts, axis=2)

        diff = tf.subtract(offset_pred, offset_gt)
        abs_diff = tf.multiply(tf.math.abs(diff), w)
        in_loss = abs_diff
        l1_loss = tf.reduce_sum(in_loss) / num_nonzero

        return l1_loss

    @tf.function
    def call(self, y_true, y_pred):
        rt, mask = y_true[0], y_true[1]

        kp_pred, seg_pred, cp_pred = y_pred[0], y_pred[1], y_pred[2]
        xyz, sampled_inds, kpts_cpts = y_pred[3], y_pred[4], y_pred[5]

        mask_selected = tf.gather_nd(mask, sampled_inds)

        kp_gt, cp_gt = self.get_offst(
            rt,
            xyz,
            mask_selected,
            kpts_cpts,
        )

        loss_kp = self.l1_loss(
            offset_pred=kp_pred, offset_gt=kp_gt, mask_labels=mask_selected
        )

        if self.binary_loss:
            loss_seg = self.BinaryFocalLoss(
                mask_selected, seg_pred
            )  # return batch-wise value
        else:
            raise NotImplementedError
            loss_seg = self.CategoricalCrossentropy(
                label, seg_pred
            )  # labels [bs, n_pts, n_cls] this is from logits

        loss_cp = self.l1_loss(
            offset_pred=cp_pred, offset_gt=cp_gt, mask_labels=mask_selected
        )

        loss_cp = self.cp_loss_discount * loss_cp
        loss_kp = self.kp_loss_discount * loss_kp
        loss_seg = self.seg_loss_discount * loss_seg

        loss = loss_cp + loss_kp + loss_seg

        return loss
