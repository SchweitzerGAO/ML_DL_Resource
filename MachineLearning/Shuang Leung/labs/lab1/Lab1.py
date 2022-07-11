import sklearn.svm as svm
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()

from sklearn.model_selection import train_test_split

# For convenience, global variables are used here, but this is not a good practice.
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0,
                                                    test_size=0.2)


def Vis(test, pred):
    # Matplotlib Visualization 
    # Tip: Using Scatter Figure
    plt.figure((500, 500), dpi=80)
    plt.title('KNN Classifier')
    feature_1 = test[:, 0]
    feature_2 = test[:, 1]
    for i in range(len(feature_1)):
        if pred[i] == 0:
            plt.scatter(feature_1[i], feature_2[i], c='r', marker='o')
        elif pred[i] == 1:
            plt.scatter(feature_1[i], feature_2[i], c='g', marker='*')
        else:
            plt.scatter(feature_1[i], feature_2[i], c='b', marker='+')


def sub_vis(test, pred_KNN, pred_LR, pred_DT, pred_SVM):
    plt.figure(figsize=(10, 6))
    feature_1 = test[:, 0]
    feature_2 = test[:, 1]
    # KNN
    plt.subplot(221)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('KNN Classifier')

    for i in range(len(feature_1)):
        if pred_KNN[i] == 0:
            plt.scatter(feature_1[i], feature_2[i], c='r', marker='o')
        elif pred_KNN[i] == 1:
            plt.scatter(feature_1[i], feature_2[i], c='g', marker='*')
        else:
            plt.scatter(feature_1[i], feature_2[i], c='b', marker='+')

    plt.subplot(222)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('LR Classifier')
    for i in range(len(feature_1)):
        if pred_LR[i] == 0:
            plt.scatter(feature_1[i], feature_2[i], c='r', marker='o')
        elif pred_LR[i] == 1:
            plt.scatter(feature_1[i], feature_2[i], c='g', marker='*')
        else:
            plt.scatter(feature_1[i], feature_2[i], c='b', marker='+')

    plt.subplot(223)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('DT Classifier')
    for i in range(len(feature_1)):
        if pred_DT[i] == 0:
            plt.scatter(feature_1[i], feature_2[i], c='r', marker='o')
        elif pred_DT[i] == 1:
            plt.scatter(feature_1[i], feature_2[i], c='g', marker='*')
        else:
            plt.scatter(feature_1[i], feature_2[i], c='b', marker='+')

    plt.subplot(224)
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('SVM Classifier')
    for i in range(len(feature_1)):
        if pred_SVM[i] == 0:
            plt.scatter(feature_1[i], feature_2[i], c='r', marker='o')
        elif pred_SVM[i] == 1:
            plt.scatter(feature_1[i], feature_2[i], c='g', marker='*')
        else:
            plt.scatter(feature_1[i], feature_2[i], c='b', marker='+')
    plt.tight_layout()
    plt.show()


def KNN(test):
    # KNN Implementation
    model_KNN = KNeighborsClassifier()
    model_KNN.fit(X_train, y_train)
    print(model_KNN.score(X_test, y_test))
    return model_KNN.predict(test)


def KNN_KFold(X, y, n_fold=10):
    model_KNN = KNeighborsClassifier()
    score = 0
    for i in range(n_fold):
        y_test_temp = y[i]
        X_train_temp = X[0:i]
        X_train_temp = np.concatenate((X_train_temp, X[i + 1:]), axis=0)
        X_train_temp = X_train_temp.reshape((int((n_fold-1)*(150/n_fold)), 4))
        X_test_temp = X[i]
        y_train_temp = y[0:i]
        y_train_temp = np.concatenate((y_train_temp, y[i + 1:]), axis=0)
        y_train_temp = y_train_temp.reshape(int((n_fold-1)*(150/n_fold)))
        # print(X_train_temp.shape)
        # print(y_train_temp.shape)
        model_KNN.fit(X_train_temp, y_train_temp)
        score += (model_KNN.score(X_test_temp, y_test_temp) / n_fold)
    return score


def Logistic(test):
    # Logistic Regression Implementation
    model_LR = LogisticRegression(max_iter=500)
    model_LR.fit(X_train, y_train)
    print(model_LR.score(X_test, y_test))
    return model_LR.predict(test)


def DecisionTree(test):
    # Decision Tree Implementation
    model_DT = DecisionTreeClassifier()
    model_DT.fit(X_train, y_train)
    print(model_DT.score(X_test, y_test))
    return model_DT.predict(test)


def SVM(test):
    # SVM Implementation
    model_SVM = svm.SVC(C=1, kernel='linear')
    model_SVM.fit(X_train, y_train)
    print(model_SVM.score(X_test, y_test))
    return model_SVM.predict(test)


def Default(test):
    print('no algorithm!')


def my_KFold(n_fold=10):
    global X_train, y_train
    X = []
    y = []
    X.append(X_test)
    y.append(y_test)
    for i in range(n_fold - 2):
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X_train, y_train, random_state=0,
                                                                                test_size=1. / (n_fold - 1 - i))
        X.append(X_test_temp)
        y.append(y_test_temp)
        X_train = X_train_temp
        y_train = y_train_temp
    X.append(X_train)
    y.append(y_train)
    return np.array(X), np.array(y)


def Predict(Algorithm, test):
    algs = {
        'KNN': KNN,
        'Logistic': Logistic,
        'DecisionTree': DecisionTree,
        'SVM': SVM
    }
    return algs.get(Algorithm, Default)(test)


if __name__ == '__main__':
    # call predict
    # y_pred_KNN = Predict('KNN', X_test)
    # y_pred_LR = Predict('Logistic', X_test)
    # y_pred_DT = Predict('DecisionTree', X_test)
    # y_pred_SVM = Predict('SVM', X_test)
    # # call vis
    # sub_vis(X_test,y_pred_KNN,y_pred_LR,y_pred_DT,y_pred_SVM)
    X, y = my_KFold(n_fold=5)
    print(KNN_KFold(X, y, n_fold=5))
    # print(X.shape)
    # print(y.shape)
    # test = X[0:2]
    # test = np.concatenate((test,X[3:]),axis=0)
    # print(test)
    # print(test.reshape((135,4)))
