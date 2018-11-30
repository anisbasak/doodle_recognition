from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def knn_classifier(dataframe):
    x_train, x_test, y_train, y_test = train_test_split(dataframe.drawing_one_dim, dataframe.word, test_size=0.25)
    knn_clf = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)

    print("accuracy found is")
    print(accuracy_score(y_test, knn_clf.predict(x_test)))
