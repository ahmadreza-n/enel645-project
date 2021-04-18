import tensorflow as tf
import visualkeras

def get_model():
    num_classes=6
    img_size=(240, 320)
    learning_rate=1e-3
    learning_decay=1e-6
    drop_out=0.1
    nchannels=3
    kshape=(3, 3)
    base_trainable=True

    input_img = tf.keras.layers.Input(img_size + (nchannels, ))

    conv1 = tf.keras.layers.Conv2D(
        64, kshape, activation='relu', padding='same', trainable=base_trainable)(input_img)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(
        64, kshape, activation='relu', padding='same', trainable=base_trainable)(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(drop_out)(pool1)

    conv2 = tf.keras.layers.Conv2D(
        128, kshape, activation='relu', padding='same', trainable=base_trainable)(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv2D(
        128, kshape, activation='relu', padding='same', trainable=base_trainable)(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(drop_out)(pool2)

    conv3 = tf.keras.layers.Conv2D(
        256, kshape, activation='relu', padding='same', trainable=base_trainable)(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv2D(
        256, kshape, activation='relu', padding='same', trainable=base_trainable)(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(drop_out)(pool3)

    conv4 = tf.keras.layers.Conv2D(
        512, kshape, activation='relu', padding='same', trainable=base_trainable)(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv2D(
        512, kshape, activation='relu', padding='same', trainable=base_trainable)(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(drop_out)(pool4)

    conv5 = tf.keras.layers.Conv2D(
        1024, kshape, activation='relu', padding='same', trainable=base_trainable)(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2DTranspose(
        1024, kshape, activation='relu', padding='same', trainable=base_trainable)(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    up6 = tf.keras.layers.concatenate(
        [tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    up6 = tf.keras.layers.Dropout(drop_out)(up6)
    conv6 = tf.keras.layers.Conv2DTranspose(
        512, kshape, activation='relu', padding='same', trainable=base_trainable)(up6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv2DTranspose(
        512, kshape, activation='relu', padding='same', trainable=base_trainable)(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    up7 = tf.keras.layers.concatenate(
        [tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    up7 = tf.keras.layers.Dropout(drop_out)(up7)
    conv7 = tf.keras.layers.Conv2DTranspose(
        256, kshape, activation='relu', padding='same', trainable=base_trainable)(up7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Conv2DTranspose(
        256, kshape, activation='relu', padding='same', trainable=base_trainable)(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    up8 = tf.keras.layers.concatenate(
        [tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    up8 = tf.keras.layers.Dropout(drop_out)(up8)
    conv8 = tf.keras.layers.Conv2DTranspose(
        128, kshape, activation='relu', padding='same', trainable=base_trainable)(up8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    conv8 = tf.keras.layers.Conv2DTranspose(
        128, kshape, activation='relu', padding='same', trainable=base_trainable)(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)

    up9 = tf.keras.layers.concatenate(
        [tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = tf.keras.layers.Dropout(drop_out)(up9)
    conv9 = tf.keras.layers.Conv2DTranspose(
        64, kshape, activation='relu', padding='same', trainable=base_trainable)(up9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    conv9 = tf.keras.layers.Conv2DTranspose(
        64, kshape, activation='relu', padding='same', trainable=base_trainable)(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)

    out = tf.keras.layers.Conv2D(
        num_classes, (1, 1), activation='softmax')(conv9)
    model = tf.keras.Model(input_img, out)

    opt = tf.keras.optimizers.Adam(lr=learning_rate, decay=learning_decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model

model = get_model()
visualkeras.layered_view(model, to_file='model.png', type_ignore=[tf.keras.layers.BatchNormalization, tf.keras.layers.AlphaDropout])
