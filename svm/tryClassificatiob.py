import re

from sklearn.svm import SVC
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, SVMSMOTE
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

base_path_normal_train = 'Saved_Embeddings/Normal/'
base_path_abnormal_train = 'Saved_Embeddings/Abnormal/'
base_path_normal_test = 'Saved_Embeddings/Normal_test/'
base_path_abnormal_test = 'Saved_Embeddings/Abnormal_test/'
use_dimensionality_reduction = False
use_scaler = True
use_age_gender_data = False
use_oversampling = True


def load_all_image_embeddings(base_path_normal, base_path_abnormal, age_gender_data):
    embeddings = []
    labels = []
    age_list = []
    gender_list = []
    normal_age_gender_data = age_gender_data[0]
    abnormal_age_gender_data = age_gender_data[1]
    normal_id_list = list(normal_age_gender_data['ID'])
    abnormal_id_list = list(abnormal_age_gender_data['ID'])
    for k in os.listdir(base_path_normal):
        age, gender = None, None
        id_cleaned = int(''.join(filter(str.isdigit, k)))
        if id_cleaned in normal_id_list:
            _, gender, age = list(normal_age_gender_data.loc[normal_age_gender_data['ID'] == id_cleaned].values.ravel())
        embeddings.append(np.load(base_path_normal + k))
        age_list.append(age)
        gender_list.append(gender)
        labels.append(0)
    for s in os.listdir(base_path_abnormal):
        age, gender = None, None
        id_cleaned = int(''.join(filter(str.isdigit, s)))
        if id_cleaned in abnormal_id_list:
            _, gender, age = list(
                abnormal_age_gender_data.loc[abnormal_age_gender_data['ID'] == id_cleaned].values.ravel())
        embeddings.append(np.load(base_path_abnormal + s))
        labels.append(1)
        age_list.append(age)
        gender_list.append(gender)
    return np.vstack(embeddings), np.array(labels), age_list, gender_list


def load_all_data():
    normal_train, abnormal_train, normal_test, abnormal_test = add_age_gender_data()

    X_train, labels_train, age_list_train, gender_list_train = load_all_image_embeddings(base_path_normal_train,
                                                                                         base_path_abnormal_train,
                                                                                         (normal_train, abnormal_train))
    X_test, labels_test, age_list_test, gender_list_test = load_all_image_embeddings(base_path_normal_test,
                                                                                     base_path_abnormal_test,
                                                                                     (normal_test, abnormal_test))
    return X_train, labels_train, X_test, labels_test, age_list_train, age_list_test, gender_list_train, gender_list_test


def do_scaling(X_train, X_test, method='standard'):
    if method == 'standard':
        scaler = StandardScaler()
        scaler.fit(X_train)
        print("Scaler used :- ", str("Standard Scaler"))
        return scaler.transform(X_train), scaler.transform(X_test)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        print("Scaler used :- ", str("MinMax Scaler"))
        return scaler.transform(X_train), scaler.transform(X_test)
    else:
        return 'Not a valid method'


def svm_model(X_train, Y_train, X_test, Y_test):
    kmeans = SVC(kernel='linear',gamma=100, C=.1)
    kmeans.fit(X_train, Y_train)
    pred_labels_test = kmeans.predict(X_test)
    pred_labels_train = kmeans.predict(X_train)
    print(pred_labels_test)
    print(Y_test)
    print("Train Accuracy :- ", str(accuracy_score(Y_train, pred_labels_train)))
    print("Test Accuracy :- ", str(accuracy_score(Y_test, pred_labels_test)))
    return Y_test, pred_labels_test


def grid_search_parameters(Data_X, Data_Y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000, .001],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001,10,100],
                  'kernel': ['rbf', 'linear', 'sigmoid']}
    grid = RandomizedSearchCV(SVC(), param_grid, verbose=3)
    grid.fit(Data_X, Data_Y)
    return grid.best_params_

def calculate_all_acc_parameters(predicted_labels, y_true):
    return accuracy_score(y_true, predicted_labels), f1_score(y_true, predicted_labels, average='weighted',
                                                              labels=np.unique(predicted_labels)), recall_score(y_true,
                                                                                                                predicted_labels,
                                                                                                                average='weighted',
                                                                                                                labels=np.unique(
                                                                                                                    predicted_labels)), precision_score(
        y_true, predicted_labels, average='weighted', labels=np.unique(predicted_labels))

def ten_cross_validation(data_X, data_Y, params,model_name='svm', folds=10):
    Accuracy, F1_score, Recall, Precision = [], [], [], []
    kfold = KFold(folds, random_state=1, shuffle=True)
    count = 1
    Fold_Number = []
    for train_index, test_index in kfold.split(data_X):
        X_train, X_test = data_X[train_index], data_X[test_index]
        y_train, y_test = data_Y[train_index], data_Y[test_index]
        svm_classifier = SVC().set_params(**params)
        svm_classifier.fit(X_train, y_train)
        accuracy, f1score, recall, precision = calculate_all_acc_parameters(svm_classifier.predict(X_test), y_test)
        Accuracy.append(accuracy)
        F1_score.append(f1score)
        Recall.append(recall)
        Precision.append(precision)
        Fold_Number.append('Fold' + str(count))
        count = count + 1
    print('Accuracy', np.mean(Accuracy))
    print('F1 Score', np.mean(F1_score))
    print('Recall', np.mean(Recall))
    print('Precision', np.mean(Precision))
    return [Fold_Number, Accuracy, F1_score, Recall, Precision]

def bbc_model(X_train, Y_train, X_test, Y_test):
    bbc = BalancedBaggingClassifier(random_state=42)
    bbc.fit(X_train, Y_train)
    pred_labels_test = bbc.predict(X_test)
    pred_labels_train = bbc.predict(X_train)
    print(pred_labels_test)
    print(Y_test)
    print("Train Accuracy :- ", str(accuracy_score(Y_train, pred_labels_train)))
    print("Test Accuracy :- ", str(accuracy_score(Y_test, pred_labels_test)))
    return Y_test, pred_labels_test


def do_dimensionality_reduction(X_train, X_test, Y_train, Y_test, method='kernelpca'):
    if method == 'kernelpca':
        earlier_dimension = X_train.shape[1]
        kernelpca = KernelPCA()
        kernelpca.fit(X_train)
        X_train = kernelpca.transform(X_train)
        X_test = kernelpca.transform(X_test)
        print("Dimenion Reduced from :- ", str(earlier_dimension), " to :- ", str(X_train.shape[1]), " Using :- ",
              str(method))
        return X_train, X_test, Y_train, Y_test
    elif method == 'pca':
        pca = PCA()
        pca.fit(X_train)
        earlier_dimension = X_train.shape[1]
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        print("Dimenion Reduced from :- ", str(earlier_dimension), " to :- ", str(X_train.shape[1]), " Using :- ",
              str(method))
        return X_train, X_test, Y_train, Y_test
    elif method == 'svd':
        svd = TruncatedSVD()
        svd.fit(X_train)
        earlier_dimension = X_train.shape[1]
        X_train = svd.transform(X_train)
        X_test = svd.transform(X_test)
        print("Dimenion Reduced from :- ", str(earlier_dimension), " to :- ", str(X_train.shape[1]), " Using :- ",
              str(method))
        return X_train, X_test, Y_train, Y_test
    elif method == 'tsne':
        earlier_dimension = X_train.shape[1]
        X = np.concatenate([X_train, X_test], axis=0)
        Y = np.concatenate([Y_train, Y_test], axis=0)
        X_embedded = TSNE(n_components=3).fit_transform(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X_embedded, Y, test_size=.1)
        print("Dimenion Reduced from :- ", str(earlier_dimension), " to :- ", str(X_train.shape[1]), " Using :- ",
              str(method))
        return X_train, X_test, Y_train, Y_test

def use_oversampling_method(X_train, Y_train, method='smote'):
    if method == 'smote':
        oversample = SMOTE()
        X_train, Y_train = oversample.fit_resample(X_train, Y_train)
        print("oversampling used :- ", str(method))
        return X_train, Y_train
    elif method == 'randomsampler':
        random = RandomOverSampler()
        X_train, Y_train = random.fit_resample(X_train, Y_train)
        print("oversampling used :- ", str(method))
        return X_train, Y_train
    elif method == 'adasyn':
        adasyn = ADASYN()
        X_train, Y_train = adasyn.fit_resample(X_train, Y_train)
        print("oversampling used :- ", str(method))
        return X_train, Y_train
    elif method == "svmsmote":
        svmsmote = SVMSMOTE()
        X_train, Y_train = svmsmote.fit_resample(X_train, Y_train)
        print("oversampling used :- ", str(method))
        return X_train, Y_train
    else:
        print('Wrong Method')



def add_age_gender_data():
    normal_train = pd.read_excel('age and gender.xlsx', sheet_name=0)
    abnormal_train = pd.read_excel('age and gender.xlsx', sheet_name=1)
    normal_test = pd.read_excel('age and gender.xlsx', sheet_name=2)
    abnormal_test = pd.read_excel('age and gender.xlsx', sheet_name=3)
    return normal_train, abnormal_train, normal_test, abnormal_test


def add_age_gender_to_data(data, agedata, gender_data):
    temp_array = []
    for index, k in enumerate(data):
        temp_k = list(k) + [agedata[index], gender_data[index]]
        temp_array.append(temp_k)
    return np.array(temp_array)


X_train, Y_train, X_test, Y_test, age_list_train, age_list_test, gender_list_train, gender_list_test = load_all_data()



print(X_train.shape)

if use_oversampling:
    X_train, Y_train = use_oversampling_method(X_train, Y_train, method='svmsmote')

print(X_train.shape)

if use_scaler:
    X_train, X_test = do_scaling(X_train, X_test)

if use_dimensionality_reduction:
    X_train, X_test, Y_train, Y_test = do_dimensionality_reduction(X_train, X_test, Y_train, Y_test, method='pca')


train_array = []

if use_age_gender_data:
    X_train = add_age_gender_to_data(X_train, age_list_train, gender_list_train)
    X_test = add_age_gender_to_data(X_test, age_list_test, gender_list_test)
    print("After Age Data added shape of Train Data ", X_train.shape)
    print("After Age Data added shape of Test Data ", X_test.shape)


result =svm_model(X_train, Y_train, X_test, Y_test)

pd.DataFrame(result).to_csv('imbalanced_svm_Model.csv', index=False)


best_parameters = grid_search_parameters(X_train, Y_train)
X = np.concatenate([X_train, X_test], axis=0)
Y = np.concatenate([Y_train, Y_test], axis=0)

print(best_parameters)
pd.DataFrame(np.array(ten_cross_validation(X, Y,best_parameters, folds=10)).T,
             columns=['Fold_Number', 'Accuracy', 'F1_Score', 'Recall', 'Precision']).to_csv('Results.csv', index=False)