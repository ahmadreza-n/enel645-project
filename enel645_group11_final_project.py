#%% Import Libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%% Load and prepare the data
INPUT_DIR = '/home/ahmadreza.nazari/train/image/'
TARGET_DIR = '/home/ahmadreza.nazari/train/label/'

IMG_SIZE = (240, 320)
N_CHANNELS = 3
CLASSES = {
    0: 'Ball',
    1: 'Field',
    2: 'Robots',
    3: 'Lines',
    4: 'Background',
    5: 'Goals',
}

num_classes = len(CLASSES)
print('Number of classes:', num_classes)

input_paths = sorted(
    [
        os.path.join(INPUT_DIR, fname)
        for fname in os.listdir(INPUT_DIR)
        if fname.endswith('.png')
    ]
)
print(f'Found {len(input_paths)} images.')

target_paths = sorted(
    [
        os.path.join(TARGET_DIR, fname)
        for fname in os.listdir(TARGET_DIR)
        if fname.endswith('.png')
    ]
)
print(f'Found {len(target_paths)} masks.')

if (len(input_paths) != len(target_paths)):
    raise Exception('dataset errror')

# Shuffle the data
indexes = np.arange(len(input_paths), dtype=int)
np.random.shuffle(indexes)

input_paths = [input_paths[i] for i in indexes]
target_paths = [target_paths[i] for i in indexes]

print('Paths: ')
for input_path, target_path in zip(input_paths[:5], target_paths[:5]):
    print(input_path, '|', target_path)

mapping = {
    (31, 120, 180): 0,
    (106, 176, 25): 1,
    (156, 62, 235): 2,
    (255, 255, 255): 3,
    (69, 144, 232): 4,
    (227, 26, 28): 5,
}


class DataSequence(tf.keras.utils.Sequence):
    '''Helper to iterate over the data as Numpy arrays.'''

    def __init__(self, in_paths, out_paths, img_size=IMG_SIZE, n_channels=N_CHANNELS, n_classes=num_classes):
        self.input_paths = in_paths
        self.target_paths = out_paths
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        '''Returns tuple (input, target) correspond to batch #idx.'''
        i = idx
        path = self.input_paths[i]
        target_path = self.target_paths[i]

        img = tf.keras.preprocessing.image.load_img(path, color_mode='rgb', target_size=self.img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255.0

        mask = tf.keras.preprocessing.image.load_img(target_path, color_mode='rgb', target_size=self.img_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        # replace colors with corresponding labels
        result = np.ndarray(shape=mask.shape[:2], dtype='float32')
        result[:, :] = -1
        for rgb, idx in mapping.items():
            result[(mask == rgb).all(2)] = idx
        result[(mask == (0, 0, 0)).all(2)] = 1
        if(result.min() == -1):
            colors = set(tuple(v) for m2d in mask for v in m2d)
            # document incorrect mapping
            print('\nincorrect mapping')
            print(colors)
        # One-hot encoded representation
        result = tf.keras.utils.to_categorical(result, self.n_classes)

        return img, result


# Split dataset into train/validation/test
val_samples = int(0.15 * len(input_paths))
test_samples = int(0.05 * len(input_paths))
train_samples = len(input_paths) - val_samples - test_samples

train_input_paths = input_paths[:train_samples]
train_target_paths = target_paths[:train_samples]

val_input_paths = input_paths[train_samples:train_samples + val_samples]
val_target_paths = target_paths[train_samples:train_samples + val_samples]

test_input_paths = input_paths[train_samples +
                               val_samples: train_samples + val_samples + test_samples]
test_target_paths = target_paths[train_samples +
                                 val_samples:train_samples + val_samples + test_samples]

train_gen = DataSequence(train_input_paths, train_target_paths)
val_gen = DataSequence(val_input_paths, val_target_paths)
test_gen = DataSequence(test_input_paths, test_target_paths)

print('simulation data train_samples', train_samples)
print('simulation data val_samples', val_samples)
print('simulation data test_samples', test_samples)


#%% Define the model
# weight of each class in the whole dataset
weights = np.array([1-0.008129217, 1-0.741364343, 1-0.038759669,
                    1-0.033972565, 1-0.159647414, 1-0.018480072])


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


ALPHA = 0.8
GAMMA = 2


def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = tf.keras.backend.flatten(inputs)
    targets = tf.keras.backend.flatten(targets)
    BCE = tf.keras.backend.binary_crossentropy(targets, inputs)
    BCE_EXP = tf.keras.backend.exp(-BCE)
    focal_loss = tf.keras.backend.mean(
        alpha * tf.keras.backend.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss


def jaccard_coef(y_true, y_pred, smooth=1):
    intersection = tf.keras.backend.sum(
        tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(
        y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac * smooth


def jaccard_loss(y_true, y_pred, smooth=1):
    return (smooth - jaccard_coef(y_true, y_pred, smooth))


def weighted_dice_loss(y_true, y_pred):
    smooth = 1.
    w, m1, m2 = weights * weights, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
            (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
    loss = 1. - tf.reduce_sum(score)
    return loss


def weighted_dice_coef(y_true, y_pred):
    smooth = 1.
    w, m1, m2 = weights * weights, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * tf.reduce_sum(w * intersection) + smooth) / \
            (tf.reduce_sum(w * m1) + tf.reduce_sum(w * m2) + smooth)
    return tf.reduce_sum(score)


def get_unet_mod(num_classes=num_classes, img_size=IMG_SIZE, learning_rate=1e-3,
                 learning_decay=1e-6, drop_out=0.1, nchannels=N_CHANNELS, kshape=(3, 3),
                 base_trainable=True):

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
    model.compile(optimizer=opt, loss=jaccard_loss,
                  metrics=[dice_coef, jaccard_coef, weighted_dice_coef, 'accuracy'])

    return model


#%% Build model
tf.keras.backend.clear_session()
model = get_unet_mod()
model.summary()

model.save_weights('/home/ahmadreza.nazari/unet_seg_v12.h5')
model_name = 'unet_seg_v12.h5'
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True,
                                             save_weights_only=True, mode='min')
# Learning rate schedule


def scheduler(epoch, lr):
    if epoch % 3 == 0 and epoch != 0:
        lr = lr/2
    return lr


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

callbacks = [early_stop, monitor, lr_schedule]


#%% Train the model, doing validation at the end of each epoch.
epochs = 7
history = model.fit(train_gen, epochs=epochs,
                    validation_data=val_gen, callbacks=callbacks)

model.load_weights('/home/ahmadreza.nazari/unet_seg_v12.h5')

#%% Extract Metrics
print('Evaluating with simulation data...')
metrics = model.evaluate(test_gen)
for i, metric in enumerate(metrics):
    print(f'Metric {i}:', metric)

#%% Visualize Test Results
print('Getting predictions...')

test_preds = model.predict(test_gen)
palette = {value: key for (key, value) in mapping.items()}

plt.figure(figsize=(8, 20), dpi=300)
plt_number = 10
indexes = np.arange(len(test_input_paths), dtype=int)
np.random.shuffle(indexes)
indexes = indexes[:plt_number]
for j, i in enumerate(indexes):
    img = plt.imread(test_input_paths[i])
    plt.subplot(plt_number, 3, j*3+1)
    plt.imshow(img/img.max())
    plt.axis('off')
    plt.title('Image')
    plt.subplot(plt_number, 3, j*3+2)
    predicted_mask = np.argmax(test_preds[i], axis=-1)

    predicted_mask_rgb = np.ndarray(
        shape=predicted_mask.shape + (3, ), dtype=float)
    for key in palette.keys():
        predicted_mask_rgb[predicted_mask == key] = palette[key]
    predicted_mask_rgb /= 255.0

    plt.imshow(predicted_mask_rgb)
    plt.axis('off')
    plt.title('Predicted Mask')
    plt.subplot(plt_number, 3, j*3+3)
    actual_mask = tf.keras.preprocessing.image.load_img(
        test_target_paths[i], color_mode='rgb', target_size=IMG_SIZE)
    actual_mask = tf.keras.preprocessing.image.img_to_array(actual_mask)
    plt.imshow(actual_mask/actual_mask.max())
    plt.axis('off')
    plt.title('Actual Mask')
plt.savefig('/home/ahmadreza.nazari/test-results.png')

print('Done!')
