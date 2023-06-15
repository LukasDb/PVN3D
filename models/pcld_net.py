import tensorflow as tf
from .pointnet2_tf import _PointNet2TfModel, PointNet2Params
from .pointnet2 import _PointNetModel
from .densefusion import _DenseFusionNet, DenseFusionNetParams
from .mlp import _MlpNets, MlpNetsParams


class PcldNetNetParams:
    def __init__(self):
        self.point_net2_params = PointNet2Params()
        self.dense_fusion_params = DenseFusionNetParams()
        self.mlp_params = MlpNetsParams()


class _PcldNet(tf.keras.Model):
    def __init__(
        self, params: PcldNetNetParams, num_pts, num_kpts, num_cls, num_cpts, dim_xyz
    ):
        super(_PcldNet, self).__init__()
        self.params = params

        if self.params.point_net2_params.use_tf_interpolation:
            self.pointnet2_model = _PointNet2TfModel(
                self.params.point_net2_params, num_classes=num_cls
            )
        else:
            self.pointnet2_model = _PointNetModel(
                self.params.point_net2_params, num_classes=num_cls
            )

        self.dense_fusion_net = _DenseFusionNet(self.params.dense_fusion_params)

        self.dense_model = self.dense_fusion_net.build_dense_model(
            pcl_emb_shape=(num_pts, self.params.dense_fusion_params.num_embeddings)
        )

        self.mlp_net = _MlpNets(
            self.params.mlp_params,
            num_pts=num_pts,
            num_kpts=num_kpts,
            num_cls=num_cls,
            num_cpts=num_cpts,
            channel_xyz=dim_xyz,
        )

        self.num_rgbd_feats = list(self.dense_model.output_shape)[-1]

        self.mlp_model = self.mlp_net.build_mlp_model(
            rgbd_features_shape=(num_pts, self.num_rgbd_feats)
        )

    def call(self, pcld, training=None, mask=None):
        pcld_emb = self.pointnet2_model(pcld, training=training)
        dense_feats = self.dense_model(pcld_emb, training=training)
        kp, sm, cp = self.mlp_model(dense_feats, training=training)

        return kp, sm, cp
