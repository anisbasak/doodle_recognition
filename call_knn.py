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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from matplotlib.cbook import get_sample_data

#ffe6e6, #ff3333
#ccffdd, #00cc44

type = ['apple', 'hammer', 'ladder', 'suitcase', 'sun', 'table', 'tree', 'triangle', 'umbrella', 'vase']



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    print('X', X)

    cmap_light = ListedColormap(['tan', 'dimgrey', 'darkorchid', 'lightgreen', 'red', 'paleturquoise', 'darkorange','peru', 'hotpink'])
    cmap_bold = ListedColormap(['moccasin', 'lightgrey', 'plum', 'darkseagreen', 'lightsalmon', 'lightseagreen', 'navajowhite', 'wheat', 'pink'])
    # plot the decision surface
    # image_path = get_sample_data('knn.png')
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    print("y", len(y))
    plt.scatter(X[:, 0], X[:, 1], c=np.array(y), cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.title("3-Class classification (k = %i, weights = '%s')"
    #           % (n_neighbors, weights))
    plt.ylabel('Principle Component 1')
    plt.xlabel('Principle Component 2')
    plt.title('KNN Plot')
    plt.show()


def generate_graph(X, y, h = 0.2):
    X = np.array(X)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Put the result into a color plot
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Data points")
    plt.show()

def data_converter(array_draw, dimension):
    pixel_two_dim_array = np.zeros((dimension,dimension))
    for i, array in enumerate(array_draw):
        for i in range(len(array[0])):
            pixel_two_dim_array[array[0][i]//32][array[1][i]//32] = 1.0

    pixels_one_dim_array = np.array(pixel_two_dim_array).flatten().tolist()

    return pixels_one_dim_array


def shuffle_dataframe(dataframe):
    df = shuffle(dataframe).reset_index(drop=True)
    return df


def attach_one_dim_drawing(df):
    df['drawing_one_dim'] = df.apply(lambda row: data_converter(row['drawing'], 16), axis=1)
    return df


def attach_word(df):
    df['word_enum'] = df.apply(lambda row: type.index(row['word']), axis=1)
    return df


def knn_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.iloc[:, 0:2], dataframe.target, test_size=0.25)
    sc = StandardScaler()
    sc.fit(x_train)
    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)
    knn_clf = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    print('accuracy knn:', accuracy_score(y_test, knn_clf.predict(x_test)))
    print('actual:', y_test.tolist()[0])
    print('actual:', y_test.tolist())
    print('predict:', knn_clf.predict(x_test.iloc[0].values.reshape(1, -1)))
    filename = 'knn_model.sav'
    # pickle.dump(knn_clf, open(filename, 'wb'))

    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    plot_decision_regions(X_test_std, y_test, knn_clf)


def svm_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.iloc[:, 0:2], dataframe.target, test_size=0.25)
    sc = StandardScaler()
    sc.fit(x_train)
    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)
    svm_clf = svm.SVC(kernel='rbf').fit(x_train,y_train)
    y_pred = svm_clf.predict(x_test)
    x_test = pd.DataFrame(x_test)
    x_train = pd.DataFrame(x_train)
    print('test_shape',x_test.shape)
    print('train_shape',x_train.shape[1])
    print("Classification report for classifier %s:\n%s\n", (svm_clf, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s", metrics.confusion_matrix(y_test, y_pred))
    print('accuracy svm:', accuracy_score(y_test, svm_clf.predict(x_test)))
    print('actual:', len(y_test.tolist()))
    print('actual:', y_test.tolist())
    print('predict:', svm_clf.predict(x_test.iloc[0].values.reshape(1, -1)))
    # filename = 'svm_model.sav'
    # pickle.dump(svm_clf, open(filename, 'wb'))
    #
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    plot_decision_regions(X_test_std, y_test, svm_clf)

def kmeans_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.iloc[:, 0:2], dataframe.target, test_size=0.25)
    sc = StandardScaler()
    sc.fit(x_train)
    X_train_std = sc.transform(x_train)
    X_test_std = sc.transform(x_test)
    kmeans_clf = KMeans(n_clusters=2).fit(x_train)
    y_pred = kmeans_clf.predict(x_test)
    x_test = pd.DataFrame(x_test)
    x_train = pd.DataFrame(x_train)
    print('test_shape',x_test.shape)
    print('train_shape',x_train.shape[1])
    print("Classification report for classifier %s:\n%s\n", (kmeans_clf, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s", metrics.confusion_matrix(y_test, y_pred))
    print('accuracy svm:', accuracy_score(y_test, kmeans_clf.predict(x_test)))
    print('actual:', len(y_test.tolist()))
    print('actual:', y_test.tolist())
    print('predict:', kmeans_clf.predict(x_test.iloc[0].values.reshape(1, -1)))
    # filename = 'svm_model.sav'
    # pickle.dump(svm_clf, open(filename, 'wb'))
    #
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(x_test, y_test)
    plot_decision_regions(X_test_std, y_test, kmeans_clf)


def combine_data():
    path_to_json = 'rawdata/'
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
                if i == 500:
                    break


    shuffle_df = shuffle_dataframe(df)
    final_df = attach_word(attach_one_dim_drawing(shuffle_df))

    # Make features
    # columns_drawing = ['x{}'.format(i) for i in range(len(final_df.drawing_one_dim[0]))]
    columns_drawing = ['x{}'.format(i) for i in range(len(final_df.drawing_one_dim[0]))]
    print(len(final_df.drawing_one_dim[0]))
    print(columns_drawing)
    final_df[columns_drawing] = pd.DataFrame(final_df.drawing_one_dim.values.tolist(), index=final_df.index)


    print(final_df.shape)
    # # final_df.to_hdf('final.hdf', key='final_df', mode='w')
    # # # final_df.to_pickle('final.pkl')
    # svm_classifier(final_df)

    # pca = PCA(n_components=256)
    # print(pca)
    # # number of features vs explained variance graph.
    # pca.fit(final_df[columns_drawing])
    # variance = pca.explained_variance_ratio_
    # var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('# of Features')
    # plt.title('PCA Analysis')
    # # plt.ylim(30, 100.5)
    # plt.style.context('seaborn-whitegrid')
    # plt.plot(var)
    # plt.show()
    # plt.savefig('features_vs_var')
    #
    # # Explained variance vs PC
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('Principal components')
    # plt.title('PCA Analysis')
    # # plt.ylim(30, 100.5)
    # plt.style.context('seaborn-whitegrid')
    # plt.plot(pca.explained_variance_ratio_)
    # plt.show()
    # plt.savefig('pca_vs_var')

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(final_df[columns_drawing])
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = final_df['word_enum']
    print(principalDf.shape)
    print(principalDf.dropna())
    # knn_classifier(principalDf.dropna())
    # svm_classifier(principalDf)
    kmeans_classifier(principalDf)

def apply_knn():
    df = pd.read_hdf('final.hdf', 'final_df')
    print(df.shape)
    knn_classifier(df)


combine_data()

# apply_knn()
