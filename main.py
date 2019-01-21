import os

from PIL import Image
import tensorflow as tf
from tensorflow.python.keras import models, layers, losses, optimizers
from tensorflow.python.keras.applications import MobileNetV2

tf.enable_eager_execution()

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


def create_datasets(background_dir, closed_dir, open_dir, val_train_split):
    bg_list = os.listdir(background_dir)
    cl_list = os.listdir(closed_dir)
    op_list = os.listdir(open_dir)
    n_total = len(bg_list + cl_list + op_list)

    def image_generator(img_list, img_dir):
        for img_name in img_list:
            yield Image.open(os.path.join(img_dir, img_name))

    bg_data = tf.data.Dataset().from_generator(lambda: image_generator(bg_list, background_dir), tf.uint8)
    cl_data = tf.data.Dataset().from_generator(lambda: image_generator(cl_list, closed_dir), tf.uint8)
    op_data = tf.data.Dataset().from_generator(lambda: image_generator(op_list, open_dir), tf.uint8)

    def parse_data_map_func(element, finger_status, face_status):
        x = element
        finger_status = tf.one_hot(finger_status, 3)
        finger_positions = [0, 0, 0, 0]
        face_positions = [0, 0, 0, 0]
        return x, (finger_status, finger_positions, face_status, face_positions)

    bg_data = bg_data.map(lambda el: parse_data_map_func(el, 0, 0))
    cl_data = cl_data.map(lambda el: parse_data_map_func(el, 1, 1))
    op_data = op_data.map(lambda el: parse_data_map_func(el, 2, 1))

    all_data = bg_data.concatenate(cl_data).concatenate(op_data).shuffle(n_total)
    n_train = int(val_train_split * n_total)
    n_val = n_total-n_train
    train_data = all_data.take(n_train)
    val_data = all_data.skip(n_train)
    return train_data, n_train, val_data, n_val


def main():

    train_data, steps_per_epoch, val_data, val_steps = create_datasets("./data/background",
                                                                       "./data/closed",
                                                                       "./data/open")

    model = construct_model()
    model.summary()

    loss = ['categorical_crossentropy',
            'mean_absolute_error',
            'binary_crossentropy',
            'mean_absolute_error']

    model.compile(optimizers.Adam(),
                  loss=loss)


if __name__ == "__main__":
    train_data, steps_per_epoch, val_data, val_steps = create_datasets("./data/background",
                                                                       "./data/closed",
                                                                       "./data/open",
                                                                       0.1)

    i = 0
    for x, y in train_data:
        i += 1
        img = Image.fromarray(x.numpy())
        img.show()
        if i == 10:
            break
