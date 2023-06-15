import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from .resnet import ResNetParams, _ResNet
from .pspnet import PspNetParams, _PspNet
from .pointnet2_tf import _PointNet2TfModel, PointNet2Params
from .pointnet2 import _PointNetModel
from .densefusion import _DenseFusionNet, DenseFusionNetParams
from .mlp import _MlpNets, MlpNetsParams
from .utils import match_choose_adp
from typing import List, Dict



class Pvn3dNetParams:
    def __init__(self):
        self.resnet_params = ResNetParams()
        self.psp_params = PspNetParams()
        self.point_net2_params = PointNet2Params()
        self.dense_fusion_params = DenseFusionNetParams()
        self.mlp_params = MlpNetsParams()


class PVN3D(tf.keras.Model):
    resnet_params: ResNetParams
    psp_params: PspNetParams
    point_net2_params: PointNet2Params
    dense_fusion_params: DenseFusionNetParams
    mlp_params: MlpNetsParams

    resnet_input_shape: List[int]
    num_kpts: int
    num_cls: int
    num_cpts: int
    dim_xyz: int

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
        build = True,
    ):
        super().__init__()
        self.resnet_input_shape = resnet_input_shape
        self.num_kpts = num_kpts
        self.num_cls = num_cls
        self.num_cpts = num_cpts
        self.dim_xyz = dim_xyz
        self.resnet_params = ResNetParams(**resnet_params)
        self.psp_params = PspNetParams(**psp_params)
        self.point_net2_params = PointNet2Params(**point_net2_params)
        self.dense_fusion_params = DenseFusionNetParams(**dense_fusion_params)
        self.mlp_params = MlpNetsParams(**mlp_params)

        self.num_pts = num_pts = self.point_net2_params.n_sample_points

        self.resnet_net = _ResNet(self.resnet_params, self.resnet_input_shape)
        self.resnet_model = self.resnet_net.build_resnet_model()

        self.psp_net = _PspNet(self.psp_params)
        self.psp_model = self.psp_net.build_psp_model(
            list(self.resnet_model.output_shape)[1:]
        )

        if self.point_net2_params.use_tf_interpolation:
            self.pointnet2_model = _PointNet2TfModel(
                self.point_net2_params, num_classes=self.num_cls
            )
        else:
            raise NotImplementedError

        self.dense_fusion_net = _DenseFusionNet(self.dense_fusion_params)
        self.dense_fusion_model = self.dense_fusion_net.build_dense_fusion_model(
            rgb_emb_shape=(num_pts, self.dense_fusion_params.num_embeddings),
            pcl_emb_shape=(num_pts, self.dense_fusion_params.num_embeddings),
        )

        self.mlp_net = _MlpNets(
            self.mlp_params,
            num_pts=num_pts,
            num_kpts=self.num_kpts,
            num_cls=self.num_cls,
            num_cpts=self.num_cpts,
            channel_xyz=self.dim_xyz,
        )

        self.num_rgbd_feats = list(self.dense_fusion_model.output_shape)[-1]
        self.mlp_model = self.mlp_net.build_mlp_model(
            rgbd_features_shape=(num_pts, self.num_rgbd_feats)
        )

        if build:
            n_samples = self.point_net2_params.n_sample_points
            h, w = self.resnet_input_shape[:2]
            bs = 5
            pcld_xyz = tf.zeros((bs, n_samples, 3))
            pcld_feats = tf.zeros((bs, n_samples, 6))
            rgb = tf.zeros((bs, h, w, 3))
            sampled_index = tf.zeros((bs, n_samples), tf.int32)
            crop_factor = tf.ones((bs,), tf.int32)
            self.call([[pcld_xyz, pcld_feats], sampled_index, rgb, crop_factor])


    def call(self, inputs, training=None, mask=None):
        (
            pcld,
            sampled_index,
            rgb,
            crop_factor,
        ) = inputs  # resized for resnet -> crop factor

        feats = self.resnet_model(rgb, training=training)
        pcld_emb = self.pointnet2_model(pcld, training=training)
        rgb_features = self.psp_model(feats, training=training)
        rgb_emb = match_choose_adp(
            rgb_features, sampled_index, crop_factor, self.resnet_input_shape
        )
        feats_fused = self.dense_fusion_model([rgb_emb, pcld_emb], training=training)
        kp, sm, cp = self.mlp_model(feats_fused, training=training)
        return kp, sm, cp
