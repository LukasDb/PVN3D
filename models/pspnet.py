import tensorflow as tf
import keras
from keras.layers import (
    Conv2D,
    UpSampling2D,
    BatchNormalization,
    Input,
    SpatialDropout2D,
    PReLU,
    Concatenate,
    ReLU,
)
from tensorflow_addons.layers import AdaptiveAveragePooling2D
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PspNetParams:
    build_seg_model: bool
    if_use_dropout: bool

    pyramid_block_output_size: List[int]
    pyramid_conv_kernel: Tuple[int, int]
    pyramid_conv_dim: int

    feats_2_conv_dim: int

    upsample_scale: int
    upsample_conv_kernel: Tuple[int, int]
    upsample_1_conv_dim: int
    upsample_2_conv_dim: int
    upsample_3_conv_dim: int

    psp_features_dim: int
    psp_features_conv_kernel: Tuple[int, int]

    activation: str


class _PspNet:
    def __init__(self, params: PspNetParams):
        super().__init__()
        self.params = params

    def pspnet_layers(self, resnet_feats):
        sub_feats = []

        for i, size in enumerate(self.params.pyramid_block_output_size):
            feats = self.pyramid_sampling_block(
                features=resnet_feats, size=size, name="pyramid_sample_" + str(i)
            )
            sub_feats.append(feats)

        sub_feats.append(resnet_feats)

        feats_1 = Concatenate()(sub_feats)

        feats_2 = Conv2D(
            filters=self.params.feats_2_conv_dim,
            kernel_size=(1, 1),
            activation="relu",
            use_bias=False,
            name="psp_conv2D_1",
        )(feats_1)

        feats_2 = self.drop_out_block(
            input_features=feats_2,
            if_use_dropout=self.params.if_use_dropout,
            rate=0.3,
            name="psp_drop_out",
        )

        psp_up_1 = self.psp_upsample_module(
            feats_2,
            scale=self.params.upsample_scale,
            num_conv=self.params.upsample_1_conv_dim,
            name="upsample_1",
        )

        psp_up_1 = self.drop_out_block(
            input_features=psp_up_1,
            if_use_dropout=self.params.if_use_dropout,
            rate=0.3,
            name="upsample_1_dropout",
        )

        psp_up_2 = self.psp_upsample_module(
            psp_up_1,
            scale=self.params.upsample_scale,
            num_conv=self.params.upsample_2_conv_dim,
            name="upsample_2",
        )

        psp_up_2 = self.drop_out_block(
            input_features=psp_up_2,
            if_use_dropout=self.params.if_use_dropout,
            rate=0.15,
            name="upsample_2_dropout",
        )

        psp_up_3 = self.psp_upsample_module(
            psp_up_2,
            scale=self.params.upsample_scale,
            num_conv=self.params.upsample_3_conv_dim,
            name="upsample_3",
        )

        return psp_up_3

    def pyramid_sampling_block(self, features, size, name):
        """
        :param features: input feature maps
        :param size: size of features pooled
        :return: the feature maps
        """
        features_h = features.shape[1]
        features_w = features.shape[2]

        upsample_factor_h = int(features_h / size)
        upsample_factor_w = int(features_w / size)

        sub_pool_feats = AdaptiveAveragePooling2D(
            output_size=size, name=name + "_adaptive_ave_pool"
        )(features)

        sub_conv_feats = Conv2D(
            filters=self.params.pyramid_conv_dim,
            kernel_size=self.params.pyramid_conv_kernel,
            use_bias=False,
            activation="relu",
            name=name + "_conv_2D",
        )(sub_pool_feats)

        sub_feats = UpSampling2D(
            size=(upsample_factor_h, upsample_factor_w),
            interpolation="bilinear",
            name=name + "_upsample",
        )(sub_conv_feats)

        return sub_feats

    def psp_upsample_module(self, features, scale, num_conv, name):
        features_1 = UpSampling2D(
            size=scale, interpolation="bilinear", name=name + "_upsample_2D"
        )(features)

        features_2 = Conv2D(
            filters=num_conv,
            kernel_size=self.params.upsample_conv_kernel,
            padding="same",
            use_bias=False,
            name=name + "_conv_2D",
        )(features_1)

        features_3 = BatchNormalization(name=name + "_batch_normalization")(features_2)

        if self.params.activation == "PRelu":
            features_out = PReLU(name=name + "_PRelu")(features_3)
        else:
            features_out = ReLU(name=name + "_Relu")(features_3)

        return features_out

    def drop_out_block(self, input_features, if_use_dropout, rate, name):
        if if_use_dropout:
            return SpatialDropout2D(rate=rate, name=name)(input_features)
        else:
            return input_features

    def build_psp_model(self, feats_shape=(60, 80, 512)):
        res_feats_input = Input(shape=feats_shape, name="psp_feats_input")
        up_3_features = self.pspnet_layers(res_feats_input)
        features = Conv2D(
            filters=self.params.psp_features_dim,
            kernel_size=self.params.psp_features_conv_kernel,
            name="psp_conv2D_2",
        )(up_3_features)
        model = tf.keras.Model(
            inputs=res_feats_input, outputs=features, name="PSPnet"
        )

        return model
