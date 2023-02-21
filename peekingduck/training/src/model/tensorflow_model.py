# Copyright 2023 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import tensorflow as tf
from omegaconf import DictConfig

from src.model.tensorflow_base import TFModelFactory
from src.utils.tf_model_utils import set_trainable_layers

logger = logging.getLogger("TF Model")  # pylint: disable=invalid-name
logging.basicConfig(level=logging.INFO)


class TFClassificationModelFactory(TFModelFactory):
    """Generic TensorFlow image classification model."""

    @classmethod
    def create_model(cls, model_cfg: DictConfig):
        model_name = model_cfg.model_name
        input_shape = (
            int(model_cfg.image_size),
            int(model_cfg.image_size),
            3,
        )
        num_classes = model_cfg.num_classes
        prediction_layer_name = "prediction_modified"
        # dropout_rate = model_cfg.dropout_rate
        # inputs = tf.keras.Input(shape=input_shape)
        # trainable = True if model_cfg.unfreeze_layers == -1 else False

        # x = cls.create_base(model_name, input_shape, trainable)(
        #     inputs, training=False
        # )  # disable batch norm for base model
        # outputs = cls.create_head(x, num_classes, dropout_rate)
        # model = tf.keras.Model(inputs, outputs)

        model = getattr(tf.keras.applications, model_name)(
            input_shape=input_shape, include_top=True, weights="imagenet"
        )
        # exclude the existing prediction layer
        x = model.layers[-2].output
        # create the new prediction layer
        predictions = tf.keras.layers.Dense(
            num_classes, activation="softmax", name=prediction_layer_name
        )(x)
        # Create new model with modified classification layer
        model = tf.keras.Model(inputs=model.inputs, outputs=predictions)
        # freeze all the layers except the prediction layer
        set_trainable_layers(model, [prediction_layer_name])

        logger.info("model created!")
        return model

    # @classmethod
    # def create_base(cls, model_name, input_shape, trainable):
    #     base_model = getattr(tf.keras.applications, model_name)(
    #         input_shape=input_shape, include_top=False, weights="imagenet"
    #     )
    #     # To-do: allow user to unfreeze certain number of layers for fine-tuning
    #     base_model.trainable = trainable
    #     return base_model

    # @classmethod
    # def create_head(cls, inputs, num_classes, dropout_rate):
    #     # create the pooling and prediction layers
    #     global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    #     dropout = tf.keras.layers.Dropout(dropout_rate)
    #     prediction_layer = tf.keras.layers.Dense(num_classes, activation="softmax")
    #     # chain up the layers
    #     x = global_average_layer(inputs)
    #     x = dropout(x)
    #     outputs = prediction_layer(x)
    #     return outputs
