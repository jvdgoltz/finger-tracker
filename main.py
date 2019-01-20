import tensorflow as tf
from tensorflow.python.keras import models, layers, losses, optimizers
from tensorflow.python.keras.applications import MobileNetV2


def construct_model():
    conv_base = MobileNetV2(input_shape=None,
                            include_top=False,
                            weights='imagenet',
                            pooling='max')

    conv_features = conv_base.output

    finger_status = layers.Dense(3, activation='softmax', name="finger_status")(conv_features)  # none, closed, open
    finger_positions = layers.Dense(4, activation='relu', name="finger_pos")(conv_features)  # y1, x1, y2, x2
    face_status = layers.Dense(1, activation='sigmoid', name="face_status")(conv_features)  # none, present
    face_position = layers.Dense(4, activation='relu', name="face_pos")(conv_features)  # y1, x1, y2, x2

    final_outputs = [finger_status,
                     finger_positions,
                     face_status,
                     face_position]
    model = models.Model(inputs=conv_base.input, outputs=final_outputs)

    return model


def create_datasets(background_dir, open_dir, closed_dir):
    background = tf.data.Dataset().from_tensor_slices(())

    return 1 ,2, 3, 4


def main():

    train_data, steps_per_epoch, val_data, val_steps = create_datasets()

    model = construct_model()
    model.summary()

    loss = ['categorical_crossentropy',
            'mean_absolute_error',
            'binary_crossentropy',
            'mean_absolute_error']

    model.compile(optimizers.Adam(),
                  loss=loss)


if __name__ == "__main__":
    main()
