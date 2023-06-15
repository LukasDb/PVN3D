import tensorflow as tf
from tensorflow.keras.layers import Conv1D
from dataclasses import dataclass

@dataclass
class MlpNetsParams:
    kp_conv1d_1_dim: int
    kp_conv1d_2_dim: int
    kp_conv1d_3_dim: int
    cp_conv1d_1_dim: int
    cp_conv1d_2_dim: int
    cp_conv1d_3_dim: int
    seg_conv1d_1_dim: int
    seg_conv1d_2_dim: int
    seg_conv1d_3_dim: int


class _MlpNets:
    def __init__(
        self,
        params: MlpNetsParams,
        num_pts=12228,
        num_kpts=8,
        num_cls=2,
        num_cpts=1,
        channel_xyz=3,
    ):
        self.params = params
        self.num_pts = num_pts
        self.num_kpts = num_kpts
        self.num_cls = num_cls
        self.num_cpts = num_cpts
        self.channel_xyz = channel_xyz

    def kp_layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.kp_conv1d_1_dim,
            kernel_size=1,
            activation="relu",
            name="kp_conv1d_1",
        )(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.kp_conv1d_2_dim,
            kernel_size=1,
            activation="relu",
            name="kp_conv1d_2",
        )(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.kp_conv1d_3_dim,
            kernel_size=1,
            activation="relu",
            name="kp_conv1d_3",
        )(conv1d_2)
        conv1d_4 = Conv1D(
            filters=self.num_kpts * self.channel_xyz,
            kernel_size=1,
            activation=None,
            name="kp_conv1d_4",
        )(conv1d_3)

        kp_pre = tf.reshape(
            conv1d_4, shape=[-1, self.num_pts, self.num_kpts, self.channel_xyz]
        )
        return kp_pre

    def cp_layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.cp_conv1d_1_dim,
            kernel_size=1,
            activation="relu",
            name="cp_conv1d_1",
        )(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.cp_conv1d_2_dim,
            kernel_size=1,
            activation="relu",
            name="cp_conv1d_2",
        )(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.cp_conv1d_3_dim,
            kernel_size=1,
            activation="relu",
            name="cp_conv1d_3",
        )(conv1d_2)
        conv1d_4 = Conv1D(
            filters=self.num_cpts * self.channel_xyz,
            kernel_size=1,
            activation=None,
            name="cp_conv1d_4",
        )(conv1d_3)

        cp_pre = tf.reshape(
            conv1d_4, shape=[-1, self.num_pts, self.num_cpts, self.channel_xyz]
        )
        return cp_pre

    def seg_layers(self, rgbd_features):
        conv1d_1 = Conv1D(
            filters=self.params.seg_conv1d_1_dim,
            kernel_size=1,
            activation="relu",
            name="seg_conv1d_1",
        )(rgbd_features)
        conv1d_2 = Conv1D(
            filters=self.params.seg_conv1d_2_dim,
            kernel_size=1,
            activation="relu",
            name="seg_conv1d_2",
        )(conv1d_1)
        conv1d_3 = Conv1D(
            filters=self.params.seg_conv1d_3_dim,
            kernel_size=1,
            activation="relu",
            name="seg_conv1d_3",
        )(conv1d_2)
        conv1d_4 = Conv1D(
            filters=self.num_cls, kernel_size=1, activation=None, name="seg_conv1d_4"
        )(conv1d_3)

        # TODO shouldnt, the output shape be num_cls + 1 (for background??)

        return conv1d_4

    def build_mlp_model(self, rgbd_features_shape):
        input_rgbd_features = tf.keras.Input(
            shape=rgbd_features_shape, name="rgbd_features_input"
        )
        kp_pre_output = self.kp_layers(input_rgbd_features)

        sm_pre_output = self.seg_layers(input_rgbd_features)

        cp_pre_output = self.cp_layers(input_rgbd_features)
        model = tf.keras.Model(
            inputs=input_rgbd_features,
            outputs=[kp_pre_output, sm_pre_output, cp_pre_output],
            name="MLPs",
        )

        return model
