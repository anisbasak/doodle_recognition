#
# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
# import os
# import ast
# import datetime as dt
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = [16, 10]
# plt.rcParams['font.size'] = 14
# import seaborn as sns
# import cv2
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
# from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications import MobileNet
# from tensorflow.keras.applications.mobilenet import preprocess_input
# start = dt.datetime.now()
#
#
# DP_DIR = '../input/shuffle-csvs/'
# INPUT_DIR = '../input/quickdraw-doodle-recognition/'
#
# BASE_SIZE = 256
# NCSVS = 10
# NCATS = 10
# TRAINING_CATEGORIES = [('apple', 0),('hammer', 1),('ladder', 2),('suitcase', 3),('sun', 4),('table', 5),('tree', 6),('triangle', 7),('umbrella', 8),('vase', 9)]
#
# np.random.seed(seed=1987)
# tf.set_random_seed(seed=1987)
#
# def f2cat(filename: str) -> str:
#     return filename.split('.')[0]
#
# def list_all_categories():
#     files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
#     return sorted([f2cat(f) for f in files], key=str.lower)
#
#
# def apk(actual, predicted, k=3):
#     """
#     Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
#     """
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     score = 0.0
#     num_hits = 0.0
#     for i, p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i + 1.0)
#     if not actual:
#         return 0.0
#     return score / min(len(actual), k)
#
# def mapk(actual, predicted, k=3):
#     """
#     Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
#     """
#     return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
#
# def preds2catids(predictions):
#     return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])
#
# def top_3_accuracy(y_true, y_pred):
#     return top_k_categorical_accuracy(y_true, y_pred, k=3)
#
#
# STEPS = 800
# EPOCHS = 16
# size = 64
# batchsize = 100
#
#
# model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
# model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
#               metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
# print(model.summary())
#
#
#
#
# def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
#     img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
#     for t, stroke in enumerate(raw_strokes):
#         for i in range(len(stroke[0]) - 1):
#             color = 255 - min(t, 10) * 13 if time_color else 255
#             _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
#                          (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
#     if size != BASE_SIZE:
#         return cv2.resize(img, (size, size))
#     else:
#         return img
#
# def image_generator_xd(size, df, ks, lw=6, time_color=True):
#     while True:
#         df['drawing'] = df['drawing'].apply(ast.literal_eval)
#         print('--------', df['drawing'])
#         x = np.zeros((len(df), size, size, 1))
#         for i, raw_strokes in enumerate(df.drawing.values):
#             x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
#         x = preprocess_input(x).astype(np.float32)
#         y = keras.utils.to_categorical(df.y, num_classes=NCATS)
#         yield x, y
#
#
# def df_to_image_array_xd(df, size, lw=6, time_color=True):
#     df['drawing'] = df['drawing'].apply(ast.literal_eval)
#     x = np.zeros((len(df), size, size, 1))
#     for i, raw_strokes in enumerate(df.drawing.values):
#         x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
#     x = preprocess_input(x).astype(np.float32)
#     return x
#
# def change_category(word):
#     for i in range(len(TRAINING_CATEGORIES)):
#         if word == str(TRAINING_CATEGORIES[i][0]):
#             print(word)
#             return str(TRAINING_CATEGORIES[i][1])
#
#
# # Start
#
# dfArray = []
#
# for i in range(len(TRAINING_CATEGORIES)):
#     dfArray.append(pd.read_csv(str(TRAINING_CATEGORIES[i][0]) + '.csv', nrows = 100))
#
# df = pd.concat(dfArray).reset_index(drop=True)
# df = df[df['recognized'] == True]
# df = df.reset_index(drop=True)
# df['y'] = 999
#
# for i in range(len(TRAINING_CATEGORIES)):
#     for j in range(df.shape[0]):
#         if TRAINING_CATEGORIES[i][0] == df.iloc[j]['word']:
#             df.at[j, 'y'] = TRAINING_CATEGORIES[i][1]
#
# print(df.head(200))
#
# x_valid = df_to_image_array_xd(df, size)
# y_valid = keras.utils.to_categorical(df.y, num_classes=NCATS)
# print(x_valid.shape, y_valid.shape)
# print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
#
# # train_datagen = image_generator_xd(size=size, df = df, ks=range(NCSVS - 1))
# #
# # x, y = next(train_datagen)
# n = 8
# fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
# for i in range(n**2):
#     ax = axs[i // n, i % n]
#     (-x[i]+1)/2
#     ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
#     ax.axis('off')
# plt.tight_layout()
# fig.savefig('gs.png', dpi=300)
# plt.show();


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import ast
import datetime as dt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
import pickle

start = dt.datetime.now()

cwd = os.getcwd()
DP_DIR = cwd
INPUT_DIR = '..doodle_recognition/input/quickdraw-doodle-recognition/'

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    cwd = os.getcwd()
    files = os.listdir(os.path.join(cwd, 'doodle_recognition/input/train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)


def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)




STEPS = 500
EPOCHS = 5
size = 64
batchsize = 680

model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)
model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())


def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'shuffled_output/train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x


valid_df = pd.read_csv(os.path.join(DP_DIR, 'shuffled_output/train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
x_valid = df_to_image_array_xd(valid_df, size)
y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
print(x_valid.shape, y_valid.shape)
print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))

x, y = next(train_datagen)
n = 8
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x[i]+1)/2
    ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
plt.show();


callbacks = [ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5, min_delta=0.005, mode='max', cooldown=3, verbose=1)]
hists = []
hist = model.fit_generator(
    train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
    validation_data=(x_valid, y_valid),
    callbacks = callbacks
)

filename = 'mobile_net_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
hists.append(hist)


hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
hist_df.index = np.arange(1, len(hist_df)+1)
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
axs[0].plot(hist_df.val_categorical_accuracy, lw=5, label='Validation Accuracy')
axs[0].plot(hist_df.categorical_accuracy, lw=5, label='Training Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].grid()
axs[0].legend(loc=0)
axs[1].plot(hist_df.val_categorical_crossentropy, lw=5, label='Validation MLogLoss')
axs[1].plot(hist_df.categorical_crossentropy, lw=5, label='Training MLogLoss')
axs[1].set_ylabel('MLogLoss')
axs[1].set_xlabel('Epoch')
axs[1].grid()
axs[1].legend(loc=0)
fig.savefig('hist.png', dpi=300)
plt.show();


valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)
print('Map3: {:.3f}'.format(map3))