# import data-converter
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
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import pickle
from sklearn import datasets, svm, metrics


type = ['apple', 'hammer', 'ladder', 'suitcase', 'sun', 'table', 'tree', 'triangle', 'umbrella', 'vase']

def data_converter(array_draw, dimension):
    pixel_two_dim_array = np.zeros((dimension,dimension))
    for i, array in enumerate(array_draw):
        for i in range(len(array[0])):
            pixel_two_dim_array[array[0][i]][array[1][i]] = 1.0

    pixels_one_dim_array = np.array(pixel_two_dim_array).flatten().tolist()

    return pixels_one_dim_array


def shuffle_dataframe(dataframe):
    df = shuffle(dataframe).reset_index(drop=True)
    return df


def attach_one_dim_drawing(df):
    df['drawing_one_dim'] = df.apply(lambda row: data_converter(row['drawing'], 256), axis=1)
    return df


def attach_word(df):
    df['word_enum'] = df.apply(lambda row: type.index(row['word']), axis=1)
    return df


def knn_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.drawing_one_dim.tolist(), dataframe.word_enum, test_size=0.25)
    knn_clf = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)
    print('accuracy knn:', accuracy_score(y_test, knn_clf.predict(x_test)))
    filename = 'knn_model.sav'
    pickle.dump(knn_clf, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)



def svm_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.drawing_one_dim.tolist(), dataframe.word_enum, test_size=0.25)
    svm_clf = svm.SVC(gamma=0.001).fit(x_train,y_train)
    y_pred = svm_clf.predict(x_test)
    print("Classification report for classifier %s:\n%s\n", (svm_clf, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s", metrics.confusion_matrix(y_test, y_pred))
    filename = 'svm_model.sav'
    pickle.dump(svm_clf, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(x_test, y_test)


def combine_data():
    path_to_json = 'rawdata1/'
    allFiles = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.ndjson')]

    print('allFiles:', allFiles)
    df = pd.DataFrame(columns=['drawing', 'word'])
    for index, js in enumerate(allFiles):
        with open(os.path.join(path_to_json, js)) as json_file:
            for i, line in enumerate(json_file):
                json_text = json.loads(line)
                word = json_text['word']
                drawing = json_text['drawing']
                df.loc[(index+1) * i] = [drawing, word]
                print('file # {} line {}'.format(index, i))
                if i == 10000:
                    break


    shuffle_df = shuffle_dataframe(df)
    final_df = attach_word(attach_one_dim_drawing(shuffle_df))
    print(final_df.shape)
    final_df.to_pickle('final.pkl')


def apply_knn():
    df = pd.read_pickle('final.pkl')
    print(df.shape)
    knn_classifier(df)


combine_data()

apply_knn()
