import tensorflow as tf
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
from .backbone import get_backbone_model
from dataclasses import dataclass


@dataclass
class ResNetParams:
    backbone_type: str
    down_sample_factor: int
    backbone_weights: str
    interpolation_method: str
    freeze_model: bool
    include_top: bool


class _ResNet:
    def __init__(self, params: ResNetParams, input_shape):
        super().__init__()
        self.params = params
        self.input_shape = input_shape
        self.crop_factor = 1

    def build_resnet_model(self):
        input_layer = tf.keras.Input(shape=self.input_shape, name="resnet_rgb_input")
        resnet_model = get_backbone_model(
            name=self.params.backbone_type,
            input_shape=self.input_shape,
            downsample_factor=self.params.down_sample_factor,
            weights=self.params.backbone_weights,
            freeze_model=self.params.freeze_model,
            include_top=self.params.include_top,
        )

        output_features = resnet_model(input_layer)
        model = tf.keras.Model(
            inputs=input_layer, outputs=output_features, name=self.params.backbone_type
        )
        return model
