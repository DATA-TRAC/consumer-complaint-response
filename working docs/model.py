'''
*------------------*
|                  |
|     MODEL!       |
|                  |
*------------------*
'''

#------------------------------------------------------------- IMPORTS  -------------------------------------------------------------

import pandas as pd
import numpy as np
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, CategoricalNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import recall_score

#feature importance
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif



"""
*------------------*
|                  |
|    FUNCTIONS     |
|                  |
*------------------*
"""

#-------------------------------------------------------------VECTORIZERS -------------------------------------------------------------


def make_cv(Xtr,Xv,Xt):
    """
    The function `make_cv` takes in three sets of data (train, validation, and test) and converts them
    into bag-of-words representations using a CountVectorizer with n-gram range of 1 to 3, and then
    returns the transformed data as dataframes.
    
    :param Xtr: Xtr is the training data, which is a pandas DataFrame containing the lemmatized text
    data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training. It is a subset of the overall dataset that is not used for training the model but
    is used to tune the hyperparameters and assess the model's generalization ability
    :param Xt: Xt is the test data, which is a dataframe containing the text data that you want to
    classify or analyze
    :return: three dataframes: Xtr_cv, Xv_cv, and Xt_cv.
    """
    #make my bag of words up to trigrams cv and keep single characters
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=2900)
    # fit and transform train
    Xtr_bow_cv = cv.fit_transform(Xtr.lemon.astype(str))
    # transform val and test
    Xv_bow_cv = cv.transform(Xv.lemon.astype(str))
    Xt_bow_cv = cv.transform(Xt.lemon.astype(str))
    # make dfs
    Xtr_cv = pd.DataFrame(Xtr_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xtr.index)
    Xv_cv = pd.DataFrame(Xv_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xv.index)
    Xt_cv = pd.DataFrame(Xt_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xt.index)
    return Xtr_cv,Xv_cv,Xt_cv


def make_tfidf(Xtr,Xv,Xt):
    """
    The function `make_tfidf` takes in three sets of data (train, validation, and test) and applies the
    TF-IDF vectorization technique to convert the text data into numerical features, using n-grams up to
    trigrams and keeping single characters. It then returns the transformed data as pandas DataFrames.
    
    :param Xtr: Xtr is the training data, which is a dataframe containing the text data that you want to
    transform into TF-IDF features. The "lemmatized" column in the dataframe contains the preprocessed
    text data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training
    :param Xt: Xt is the input data for the test set. It is a dataframe containing the text data that
    needs to be transformed into TF-IDF representation
    :return: three dataframes: Xtr_tfidf, Xv_tfidf, and Xt_tfidf.
    """
    #make my bag of words up to trigrams tfidf and keep single characters
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=2900)
    # fit and transform train
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemon.astype(str))
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemon.astype(str))
    Xt_bow_tfidf = tfidf.transform(Xt.lemon.astype(str))
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf


# mono, bi, and trigram features
def make_mbt_cv(Xtr,Xv,Xt):
    """
    The function `make_cv` takes in three sets of data (train, validation, and test) and converts them
    into bag-of-words representations using a CountVectorizer with n-gram range of 1 to 3, and then
    returns the transformed data as dataframes.
    
    :param Xtr: Xtr is the training data, which is a pandas DataFrame containing the lemmatized text
    data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training. It is a subset of the overall dataset that is not used for training the model but
    is used to tune the hyperparameters and assess the model's generalization ability
    :param Xt: Xt is the test data, which is a dataframe containing the text data that you want to
    classify or analyze
    :return: three dataframes: Xtr_cv, Xv_cv, and Xt_cv.
    """
    #make my bag of words up to trigrams cv and keep single characters
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b',lowercase=False, max_features=2900,ngram_range=(1,3))
    # fit and transform train
    Xtr_bow_cv = cv.fit_transform(Xtr.lemon.astype(str))
    # transform val and test
    Xv_bow_cv = cv.transform(Xv.lemon.astype(str))
    Xt_bow_cv = cv.transform(Xt.lemon.astype(str))
    # make dfs
    Xtr_cv = pd.DataFrame(Xtr_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xtr.index)
    Xv_cv = pd.DataFrame(Xv_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xv.index)
    Xt_cv = pd.DataFrame(Xt_bow_cv.todense(),columns=cv.get_feature_names_out(),index=Xt.index)
    return Xtr_cv,Xv_cv,Xt_cv


def make_mbt_tfidf(Xtr,Xv,Xt):
    """
    The function `make_tfidf` takes in three sets of data (train, validation, and test) and applies the
    TF-IDF vectorization technique to convert the text data into numerical features, using n-grams up to
    trigrams and keeping single characters. It then returns the transformed data as pandas DataFrames.
    
    :param Xtr: Xtr is the training data, which is a dataframe containing the text data that you want to
    transform into TF-IDF features. The "lemmatized" column in the dataframe contains the preprocessed
    text data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training
    :param Xt: Xt is the input data for the test set. It is a dataframe containing the text data that
    needs to be transformed into TF-IDF representation
    :return: three dataframes: Xtr_tfidf, Xv_tfidf, and Xt_tfidf.
    """
    #make my bag of words up to trigrams tfidf and keep single characters
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',lowercase=False, max_features=2900,ngram_range=(1,3))
    # fit and transform train
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemon.astype(str))
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemon.astype(str))
    Xt_bow_tfidf = tfidf.transform(Xt.lemon.astype(str))
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf


def make_mbt_tfidf2(Xtr,Xv,Xt):
    """
    The function `make_tfidf` takes in three sets of data (train, validation, and test) and applies the
    TF-IDF vectorization technique to convert the text data into numerical features, using n-grams up to
    trigrams and keeping single characters. It then returns the transformed data as pandas DataFrames.
    
    :param Xtr: Xtr is the training data, which is a dataframe containing the text data that you want to
    transform into TF-IDF features. The "lemmatized" column in the dataframe contains the preprocessed
    text data
    :param Xv: Xv is the validation dataset, which is used to evaluate the performance of the model
    during training
    :param Xt: Xt is the input data for the test set. It is a dataframe containing the text data that
    needs to be transformed into TF-IDF representation
    :return: three dataframes: Xtr_tfidf, Xv_tfidf, and Xt_tfidf.
    """
    #make my bag of words up to trigrams tfidf and keep single characters
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',lowercase=False, max_features=10000, ngram_range=(1,3))
    # fit and transform train
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemon.astype(str))
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemon.astype(str))
    Xt_bow_tfidf = tfidf.transform(Xt.lemon.astype(str))
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf


def process_vector_merge(X_train,X_val,X_test):
    """
    The function `process_vector_merge` takes in three dataframes (`X_train`, `X_val`, `X_test`),
    performs some transformations on the data, and returns the merged dataframes.
    
    :param X_train: X_train is a pandas DataFrame containing the training data. It should have columns
    representing different features, with the target variable in the first column
    :param X_val: X_val is a pandas DataFrame containing the validation data. It has multiple columns,
    with the 'lemon' column being used for creating tf-idf vectors. The remaining columns are used for
    encoding
    :param X_test: X_test is a DataFrame containing the test data. It should have the same columns as
    X_train and X_val
    :return: three dataframes: X_train_tfe, X_val_tfe, and X_test_tfe.
    """
    X_train_tf, X_val_tf, X_test_tf = make_mbt_tfidf(X_train[['lemon']], X_val[['lemon']], X_test[['lemon']])
    
    encoded_train = X_train.iloc[:, 1:]
    encoded_val = X_val.iloc[:, 1:]
    encoded_test = X_test.iloc[:, 1:]
    
    X_train_tfe = encoded_train.merge(X_train_tf, left_index=True, right_index=True)
    X_val_tfe = encoded_val.merge(X_val_tf, left_index=True, right_index=True)
    X_test_tfe = encoded_test.merge(X_test_tf, left_index=True, right_index=True)
    
    # Visualization of train data
    return X_train_tfe, X_val_tfe, X_test_tfe


def process_vector_merge2(X_train,X_val,X_test):
    """
    The function `process_vector_merge2` takes three input dataframes (`X_train`, `X_val`, `X_test`),
    performs some transformations on the data, and returns three modified dataframes (`X_train_tfe`,
    `X_val_tfe`, `X_test_tfe`).
    
    :param X_train: X_train is a pandas DataFrame containing the training data. It should have columns
    representing different features or variables
    :param X_val: X_val is a pandas DataFrame containing the validation data. It has multiple columns,
    with the 'lemon' column being used for creating tf-idf vectors. The remaining columns are used for
    encoding
    :param X_test: X_test is a DataFrame containing the test data. It should have the same columns as
    X_train and X_val
    :return: three dataframes: X_train_tfe, X_val_tfe, and X_test_tfe.
    """
    X_train_tf, X_val_tf, X_test_tf = make_mbt_tfidf2(X_train[['lemon']], X_val[['lemon']], X_test[['lemon']])
    
    encoded_train = X_train.iloc[:, 1:]
    encoded_val = X_val.iloc[:, 1:]
    encoded_test = X_test.iloc[:, 1:]
    
    X_train_tfe = encoded_train.merge(X_train_tf, left_index=True, right_index=True)
    X_val_tfe = encoded_val.merge(X_val_tf, left_index=True, right_index=True)
    X_test_tfe = encoded_test.merge(X_test_tf, left_index=True, right_index=True)
    
    # Visualization of train data
    return X_train_tfe, X_val_tfe, X_test_tfe

#------------------------------------------------------------- MODELING-------------------------------------------------------------

def baseline(y_train):
    """
    The baseline function returns the percentage of the most frequent value in the y_train dataset.
    
    :param y_train: The parameter `y_train` is a pandas Series or DataFrame containing the target
    variable values for the training set
    :return: The percentage of the majority class in the y_train dataset, rounded to 4 decimal places.
    """
    print(f'Baseline Accuracy: {round(y_train.value_counts(normalize=True)[0],4)*100}%')


def process_data_modeling(train, validate, test):
    """
    The function `process_data_modeling` reads data from parquet files, performs data sampling,
    encoding, and splits the data into training, validation, and test sets.
    
    :param train: The `train` parameter is the training dataset, which is a pandas DataFrame containing
    the data for training the model
    :param validate: The `validate` parameter is a DataFrame that contains the validation data. It is
    read from a parquet file named 'validate.parquet'
    :param test: The `test` parameter is a DataFrame that contains the test data for your model. It is
    read from a parquet file named 'test.parquet'
    :return: six variables: X_train, y_train, X_val, y_val, X_test, and y_test.
    """
    response_categories = [
        'Closed with explanation',
        'Closed with non-monetary relief',
        'Closed with monetary relief',
        'Untimely response',
        'Closed'
    ]

    sm_train = []
    sm_val = []
    sm_test = []

    small_train = pd.DataFrame()
    small_val = pd.DataFrame()
    small_test = pd.DataFrame()

    random_state = 123
    percent = .2

    for category in response_categories:
        sm_train.append(int(round(len(train[train.company_response_to_consumer == category]) * percent, 0)))
        sm_val.append(int(round(len(validate[validate.company_response_to_consumer == category]) * percent, 0)))
        sm_test.append(int(round(len(test[test.company_response_to_consumer == category]) * percent, 0)))
        
        small_train = small_train.append(train[train.company_response_to_consumer == category].sample(sm_train[-1], random_state=random_state))
        small_val = small_val.append(validate[validate.company_response_to_consumer == category].sample(sm_val[-1], random_state=random_state))
        small_test = small_test.append(test[test.company_response_to_consumer == category].sample(sm_test[-1], random_state=random_state))

    small_train.reset_index(drop=True, inplace=True)
    small_val.reset_index(drop=True, inplace=True)
    small_test.reset_index(drop=True, inplace=True)

    X_train = encode(small_train)
    X_train = X_train.drop(columns=['date_received','company_response_to_consumer','state','company_name','tags','product_bins'])
    y_train = small_train['company_response_to_consumer']
    X_val = encode(small_val)
    X_val = X_val.drop(columns=['date_received','company_response_to_consumer','state','company_name','tags','product_bins'])
    y_val = small_val['company_response_to_consumer']
    X_test = encode(small_test)
    X_test = X_test.drop(columns=['date_received','company_response_to_consumer','state','company_name','tags','product_bins'])
    y_test = small_test['company_response_to_consumer']

    return X_train, y_train, X_val, y_val, X_test, y_test

def process_data_modeling2(train, validate, test):
    """
    The function `process_data_modeling` reads data from parquet files, performs data sampling,
    encoding, and splits the data into training, validation, and test sets.
    
    :param train: The `train` parameter is the training dataset, which is a pandas DataFrame containing
    the data for training the model
    :param validate: The `validate` parameter is a DataFrame that contains the validation data. It is
    read from a parquet file named 'validate.parquet'
    :param test: The `test` parameter is a DataFrame that contains the test data for your model. It is
    read from a parquet file named 'test.parquet'
    :return: six variables: X_train, y_train, X_val, y_val, X_test, and y_test.
    """
    random_state = 123
    response_categories = [
        'relief',
        'no_relief'
    ]

    sm_train = []
    sm_val = []
    sm_test = []

    small_train = pd.DataFrame()
    small_val = pd.DataFrame()
    small_test = pd.DataFrame()

    random_state = 123
    percent = .2

    for category in response_categories:
        sm_train.append(int(round(len(train[train.response == category]) * percent, 0)))
        sm_val.append(int(round(len(validate[validate.response == category]) * percent, 0)))
        sm_test.append(int(round(len(test[test.response == category]) * percent, 0)))
        
        small_train = small_train.append(train[train.response == category].sample(sm_train[-1], random_state=random_state))
        small_val = small_val.append(validate[validate.response == category].sample(sm_val[-1], random_state=random_state))
        small_test = small_test.append(test[test.response == category].sample(sm_test[-1], random_state=random_state))

    small_train.reset_index(drop=True, inplace=True)
    small_val.reset_index(drop=True, inplace=True)
    small_test.reset_index(drop=True, inplace=True)

    X_train = encode(small_train)
    X_train = X_train.drop(columns=['date_received','state','company_name','tags','product_bins', 'response'])
    y_train = small_train['response']
    X_val = encode(small_val)
    X_val = X_val.drop(columns=['date_received','state','company_name','tags','product_bins', 'response'])
    y_val = small_val['response']
    X_test = encode(small_test)
    X_test = X_test.drop(columns=['date_received','state','company_name','tags','product_bins', 'response'])
    y_test = small_test['response']

    return X_train, y_train, X_val, y_val, X_test, y_test


def log_monogram_model(Xtr,ytr,Xv,yv):
    """
    The function `log_monogram_model` trains a logistic regression model on the given training data and
    prints the train and validation accuracies.
    
    :param Xtr: The parameter Xtr represents the training data features. It is a matrix or array-like
    object that contains the features of the training data. Each row of Xtr corresponds to a sample, and
    each column corresponds to a feature
    :param ytr: The parameter "ytr" represents the target variable (labels) for the training set. It is
    a numpy array or pandas Series containing the labels corresponding to the training instances
    :param Xv: Xv is the validation set features. It is a matrix or array-like object that contains the
    input features for the validation set. Each row represents a sample, and each column represents a
    feature
    :param yv: The parameter "yv" represents the target variable for the validation set. It is the true
    labels or classes for the validation data
    """
    # unique features and tfidf
    # logistic regressor
    lr = LogisticRegression(C=0.1,random_state=123,max_iter=500)
    lr.fit(Xtr,ytr)
    # accuracies
    ytr_acc = lr.score(Xtr,ytr)
    yv_acc = lr.score(Xv,yv)
    # print results
    print('Logistic Regression Monograms')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')

def log_multigram_model(Xtr,ytr,Xv,yv):
    """
    The function `log_multigram_model` trains a logistic regression model using monograms, bigrams, and
    trigrams as features and prints the train and validation accuracies.
    
    :param Xtr: The parameter Xtr represents the training data features. It is a matrix or array-like
    object that contains the features of the training data. Each row of Xtr corresponds to a sample, and
    each column corresponds to a feature
    :param ytr: The parameter "ytr" represents the target variable (labels) for the training set. It is
    a numpy array or pandas Series containing the true labels for the training instances
    :param Xv: Xv is the feature matrix for the validation set. It contains the input features for each
    sample in the validation set
    :param yv: The parameter "yv" represents the target variable for the validation set. It is a numpy
    array or pandas Series containing the true labels for the validation set
    """
    # unique features and tfidf
    # logistic regressor
    lr = LogisticRegression(C=1,random_state=123,max_iter=500)
    lr.fit(Xtr,ytr)
    # accuracies
    ytr_acc = lr.score(Xtr,ytr)
    yv_acc = lr.score(Xv,yv)
    # print results
    print('Logistic Regression Monograms, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')

def forest_multigram_model(Xtr,ytr,Xv,yv):
    """
    The function `forest_multigram_model` trains a random forest classifier on monograms, bigrams, and
    trigrams and prints the train and validation accuracies.
    
    :param Xtr: The parameter Xtr represents the training data features. It is a matrix or array-like
    object that contains the features of the training data. Each row of Xtr corresponds to a sample, and
    each column corresponds to a feature
    :param ytr: The parameter `ytr` represents the target variable for the training set. It contains the
    true labels or classes corresponding to the input data `Xtr`
    :param Xv: The parameter Xv represents the validation set features. It is a matrix or array-like
    object that contains the input features for the validation set
    :param yv: The parameter "yv" represents the target variable for the validation set. It is a numpy
    array or pandas Series containing the true labels for the validation set
    """
    # unique features and tfidf
    # random forest
    forest = RandomForestClassifier(max_depth=7,min_samples_leaf=4,random_state=123)
    forest.fit(Xtr,ytr)
    # accuracies
    ytr_acc = forest.score(Xtr,ytr)
    yv_acc = forest.score(Xv,yv)
    # print results
    print('Random Forest Monograms, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')

def tree_multigram_model(Xtr,ytr,Xv,yv):
    """
    The function `tree_multigram_model` trains a decision tree classifier using monograms, bigrams, and
    trigrams as features and prints the train and validation accuracies.
    
    :param Xtr: Xtr is the training data features. It is a matrix or array-like object that contains the
    features of the training data. Each row represents a sample and each column represents a feature
    :param ytr: The parameter `ytr` represents the target variable for the training set. It is the set
    of labels or classes that the model will try to predict based on the features in `Xtr`
    :param Xv: Xv is the validation set features. It is a matrix or array-like object that contains the
    features of the validation set. Each row represents a sample, and each column represents a feature
    :param yv: The parameter "yv" represents the target variable for the validation set. It is the true
    labels or classes for the validation set samples
    """
    # unique features and tfidf
    # decision tree
    tree = DecisionTreeClassifier(max_depth=9,min_samples_leaf=11,random_state=123)
    tree.fit(Xtr,ytr)
    # accuracies
    ytr_acc = tree.score(Xtr,ytr)
    yv_acc = tree.score(Xv,yv)
    # print results
    print('Decision Tree Monograms, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc,4)*100}%')
    print(f'Validation Accuracy: {round(yv_acc,4)*100}%')


def test_tree(Xtr,ytr,Xt,yt):
    """
    The function `test_tree` trains a decision tree classifier on the training data `Xtr` and `ytr`, and
    then evaluates its accuracy on the test data `Xt` and `yt`.
    
    :param Xtr: The training data features (input variables) for the decision tree model
    :param ytr: The variable `ytr` represents the target variable (or the dependent variable) for the
    training set. It contains the labels or classes that we are trying to predict or classify
    :param Xt: The variable `Xt` represents the test set features. It is a matrix or dataframe
    containing the features of the test set. Each row represents an instance, and each column represents
    a feature
    :param yt: yt is the target variable for the test set. It represents the true labels or classes for
    the test set observations
    """
    # decision tree
    tree = DecisionTreeClassifier(max_depth=9,min_samples_leaf=11,random_state=123)
    tree.fit(Xtr,ytr)
    # accuracies
    yt_acc = tree.score(Xt,yt)
    # print results
    print('Decision Tree Monograms, Bigrams, Trigrams')
    print(f'Baseline Accuracy: {round(ytr.value_counts(normalize=True)[0],4)*100}%')
    print(f'Test Accuracy:     {round(yt_acc,4)*100}%')


def tree_multigram_model2(Xtr,ytr,Xv,yv):
    """
    The function `tree_multigram_model2` trains a decision tree classifier using monograms, bigrams, and
    trigrams as features and prints the train and validation accuracies.
    
    :param Xtr: Xtr is the training data features. It is a matrix or array-like object that contains the
    features of the training data. Each row represents a sample and each column represents a feature
    :param ytr: The parameter `ytr` represents the target variable for the training set. It is the set
    of labels or classes that the model will try to predict based on the features in `Xtr`
    :param Xv: Xv is the validation set features. It is a matrix or array-like object that contains the
    features of the validation set. Each row represents a sample, and each column represents a feature
    :param yv: The parameter "yv" represents the target variable for the validation set. It is the true
    labels or classes for the validation set samples
    """
    # unique features and tfidf
    # decision tree
    tree = DecisionTreeClassifier(criterion='gini',max_depth=9,min_samples_leaf=11,max_features='sqrt',random_state=123)
    tree.fit(Xtr,ytr)
    # predictions
    y_pred_train = tree.predict(Xtr)
    y_pred_val = tree.predict(Xv)
    # recall score
    tr_rec = recall_score(ytr, y_pred_train, pos_label='no_relief')
    v_rec = recall_score(yv, y_pred_val, pos_label='no_relief')
    # accuracies
    ytr_acc = tree.score(Xtr,ytr)
    yv_acc = tree.score(Xv,yv)
    # print results
    print('Decision Tree Unigrams, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc*100,2)}%')
    print(f'Validation Accuracy: {round(yv_acc*100,2)}%')
    print(f'Train Recall:        {round(tr_rec*100,2)}%')
    print(f'Validation Recall:   {round(v_rec*100,2)}%')


def svc_multigram_model(Xtr,ytr,Xv,yv):
    """
    The function `svc_multigram_model` trains a linear support vector classifier (SVC) model using
    unigrams, bigrams, and trigrams as features, and then evaluates the model's accuracy and recall
    scores on the training and validation datasets.
    
    :param Xtr: Xtr is the training data, which is a matrix or array-like object containing the features
    for training the model. Each row represents a sample, and each column represents a feature
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training data. It is
    a one-dimensional array or list containing the labels for each sample in the training set
    :param Xv: Xv is the feature matrix for the validation set. It contains the input features for each
    instance in the validation set
    :param yv: The parameter `yv` represents the true labels of the validation set. It is a list or
    array containing the actual labels for each instance in the validation set
    """
    # unique features and tfidf
    # linear svc
    sv = LinearSVC(penalty='l2',C=0.1,dual=False,random_state=123)
    sv.fit(Xtr,ytr)
    # predictions
    y_pred_train = sv.predict(Xtr)
    y_pred_val = sv.predict(Xv)
    # recall score
    tr_rec = recall_score(ytr, y_pred_train, pos_label='no_relief')
    v_rec = recall_score(yv, y_pred_val, pos_label='no_relief')
    # accuracies
    ytr_acc = sv.score(Xtr,ytr)
    yv_acc = sv.score(Xv,yv)
    # print results
    print('Linear SVC Unigrams, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc*100,2)}%')
    print(f'Validation Accuracy: {round(yv_acc*100,2)}%')
    print(f'Train Recall:        {round(tr_rec*100,2)}%')
    print(f'Validation Recall:   {round(v_rec*100,2)}%')


def mlp_multigram_model(Xtr,ytr,Xv,yv):
    """
    The function `mlp_multigram_model` trains a multi-layer perceptron model using unigrams, bigrams,
    and trigrams as features and prints the train and validation accuracies, as well as the train and
    validation recall scores.
    
    :param Xtr: Xtr is the training data features, which are the input variables used to train the MLP
    model. It should be a matrix or array-like object with shape (n_samples, n_features), where
    n_samples is the number of samples in the training data and n_features is the number of features or
    input
    :param ytr: ytr is the target variable for the training set. It contains the labels or classes
    corresponding to the input data Xtr
    :param Xv: Xv is the feature matrix for the validation set. It contains the input features for each
    instance in the validation set
    :param yv: The parameter "yv" represents the target variable for the validation set. It contains the
    true labels for the validation data
    """
    # unique features and tfidf
    # multi-layer perceptron
    ml = MLPClassifier(activation='relu',solver='adam',random_state=123,early_stopping=True)
    ml.fit(Xtr,ytr)
    # predictions
    y_pred_train = ml.predict(Xtr)
    y_pred_val = ml.predict(Xv)
    # recall score
    tr_rec = recall_score(ytr, y_pred_train, pos_label='no_relief')
    v_rec = recall_score(yv, y_pred_val, pos_label='no_relief')
    # accuracies
    ytr_acc = ml.score(Xtr,ytr)
    yv_acc = ml.score(Xv,yv)
    # print results
    print('Multi-Layer Perceptron Unigrams, Bigrams, Trigrams')
    print(f'Train Accuracy:      {round(ytr_acc*100,2)}%')
    print(f'Validation Accuracy: {round(yv_acc*100,2)}%')
    print(f'Train Recall:        {round(tr_rec*100,2)}%')
    print(f'Validation Recall:   {round(v_rec*100,2)}%')


def test_svc(Xtr,ytr,Xt,yt):
    """
    The function `test_svc` trains a linear support vector classification model on the given training
    data and evaluates its performance on the test data, printing the baseline accuracy, test accuracy,
    and test recall.
    
    :param Xtr: Xtr is the training data, which consists of the features used to train the model. It is
    a matrix or dataframe with shape (n_samples, n_features), where n_samples is the number of samples
    in the training set and n_features is the number of features
    :param ytr: The parameter `ytr` represents the target variable (labels) for the training set. It is
    the true class labels corresponding to the features `Xtr` in the training set
    :param Xt: Xt is the test set features. It represents the input data for which we want to make
    predictions
    :param yt: yt is the true labels for the test set. It is the actual target variable values for the
    test data
    """
    # linear support vector classification
    sv = LinearSVC(penalty='l2',C=0.1,dual=False,random_state=123)
    sv.fit(Xtr,ytr)
    # predictions
    y_pred_test = sv.predict(Xt)
    # recall score
    t_rec = recall_score(yt, y_pred_test, pos_label='no_relief')
    # accuracies
    yt_acc = sv.score(Xt,yt)
    # print results
    print('Linear SVC Unigrams, Bigrams, Trigrams')
    print(f'Baseline Accuracy: {round(ytr.value_counts(normalize=True)[0]*100,2)}%')
    print(f'Test Accuracy:     {round(yt_acc*100,2)}%')
    print(f'Test Recall:     {round(t_rec*100,2)}%')


#------------------------------------------------------------- LOOPING MODELs -------------------------------------------------------------



def tree_models(Xtr,ytr,Xv,yv):
    """
    Trains and evaluates decision tree models with different parameter combinations and returns the performance metrics.

    Parameters:
        Xtr (array-like): Training features.
        ytr (array-like): Training labels.
        Xv (array-like): Validation features.
        yv (array-like): Validation labels.

    Returns:
        metrics (DataFrame): A DataFrame containing performance metrics for different decision tree models.
    """
    metrics = []
# cycle through depth, leaf, class_weight for dec tree
    for d, l, cw in itertools.product(range(1, 21), range(1, 21), ['balanced', None]):
        # decision tree
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l, class_weight=cw, random_state=123)
        tree.fit(Xtr, ytr)
        # accuracies
        ytr_acc = tree.score(Xtr, ytr)
        yv_acc = tree.score(Xv, yv)
        # table-ize
        output = {
            'model': 'Decision Tree',
            'params': f"max_depth={d}, min_samples_leaf={l}, class_weight={cw}, random_state=123",
            'tr_acc': ytr_acc,
            'v_acc': yv_acc,
        }
        metrics.append(output)
    return pd.DataFrame(metrics)


def tree_models(Xtr,ytr,Xv,yv):
    """
    The function `tree_models` trains decision tree models with different combinations of
    hyperparameters and returns a DataFrame containing the training and validation accuracies for each
    model.
    
    :param Xtr: Xtr is the training data for the features (independent variables). It is a matrix or
    dataframe with shape (n_samples, n_features), where n_samples is the number of samples in the
    training set and n_features is the number of features
    :param ytr: The variable `ytr` represents the target variable (dependent variable) for the training
    set. It contains the actual values that we are trying to predict or classify
    :param Xv: Xv is the validation set features. It is a matrix or dataframe containing the input
    features for the validation set
    :param yv: The variable `yv` represents the target variable for the validation set. It is the true
    values of the target variable for the validation set
    :return: a pandas DataFrame containing the metrics for different decision tree models. The metrics
    include the model name, the parameters used for each model, the training accuracy, and the
    validation accuracy.
    """
    metrics = []
    # cycle through depth,leaf,class_weight for dec tree
    for d,l in itertools.product(range(5,11),range(5,11)):
        # decision tree
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l,random_state=123)
        tree.fit(Xtr,ytr)
        # accuracies
        ytr_acc = tree.score(Xtr,ytr)
        yv_acc = tree.score(Xv,yv)
        # table-ize
        output ={
                'model':'Decision Tree',
                'params':f"max_depth={d},min_samples_leaf={l},random_state=123",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)


def knn_models(Xtr,ytr,Xv,yv):
    """
    Trains and evaluates k-nearest neighbors (KNN) models with different parameter combinations and returns the performance metrics.

    Parameters:
        Xtr (array-like): Training features.
        ytr (array-like): Training labels.
        Xv (array-like): Validation features.
        yv (array-like): Validation labels.

    Returns:
        metrics (DataFrame): A DataFrame containing performance metrics for different KNN models.
    """
    metrics = []
    # cycle through neighbors and weights for knn
    for n,w in itertools.combinations(range(1,21),['uniform', 'distance']):
        # knn
        forest = KNeighborsClassifier(n_neighbors=n,weights=w)
        forest.fit(Xtr,ytr)
        # accuracies
        ytr_acc = forest.score(Xtr,ytr)
        yv_acc = forest.score(Xv,yv)
        # table-ize
        output ={
                'model':'KNN',
                'params':f"n_neighbors={n},weights={w}",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)


def log_models(Xtr,ytr,Xv,yv):
    """
    The function `log_models` performs logistic regression on different values of C and returns a
    DataFrame with the model parameters and accuracy scores for training and validation data.
    
    :param Xtr: Xtr is the training data features. It is a matrix or dataframe containing the input
    features for training the logistic regression model
    :param ytr: The variable `ytr` represents the target variable for the training set. It is the set of
    labels or classes that the model will try to predict based on the features in the training set `Xtr`
    :param Xv: Xv is the validation set features. It is a matrix or dataframe containing the input
    features for the validation set
    :param yv: The variable `yv` represents the target variable for the validation set. It is the true
    labels for the validation set
    :return: a pandas DataFrame containing the metrics for each logistic regression model trained with
    different values of C. The DataFrame includes the model name, the model parameters, training
    accuracy, and validation accuracy for each model.
    """
    metrics = []
    # cycle through C for log reg
    for c in [.01,.1,1,10,100,1000]:
        # logistic regression
        lr = LogisticRegression(C=c,random_state=123,max_iter=500)
        lr.fit(Xtr,ytr)
        # accuracies
        ytr_acc = lr.score(Xtr,ytr)
        yv_acc = lr.score(Xv,yv)
        # table-ize
        output ={
                'model':'LogReg',
                'params':f"C={c},random_state=123,max_iter=500",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)


def comp_nb_models(Xtr,ytr,Xv,yv):
    """
    Trains and evaluates Complement Naive Bayes models with different parameter combinations and returns the performance metrics.

    Parameters:
        Xtr (array-like): Training features.
        ytr (array-like): Training labels.
        Xv (array-like): Validation features.
        yv (array-like): Validation labels.

    Returns:
        metrics (DataFrame): A DataFrame containing performance metrics for different Complement Naive Bayes models.
    """
    # naive bayes complement
    cnb = ComplementNB(alpha=0,force_alpha=True)
    cnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = cnb.score(Xtr,ytr)
    yv_acc = cnb.score(Xv,yv)
    # table-ize
    output ={
            'model':'CNB',
            'params':f'alpha={a},force_alpha=True',
            'tr_acc':ytr_acc,
            'v_acc':yv_acc,
        }
    metrics = [output]
    # cycle through alpha for CNB
    for a in np.arange(.1,.6,.1):
        # naive bayes complement
        cnb = ComplementNB(alpha=a)
        cnb.fit(Xtr,ytr)
        # accuracies
        ytr_acc = cnb.score(Xtr,ytr)
        yv_acc = cnb.score(Xv,yv)
        # table-ize
        output ={
                'model':'CNB',
                'params':f'alpha={a}',
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)

def multi_nb_models(Xtr,ytr,Xv,yv):
    """
    Trains and evaluates Multinomial Naive Bayes models with different parameter combinations and returns the performance metrics.

    Parameters:
        Xtr (array-like): Training features.
        ytr (array-like): Training labels.
        Xv (array-like): Validation features.
        yv (array-like): Validation labels.

    Returns:
        metrics (DataFrame): A DataFrame containing performance metrics for different Multinomial Naive Bayes models.
    """
    # naive bayes multinomial
    mnb = MultinomialNB(alpha=0)
    mnb.fit(Xtr,ytr)
    # accuracies
    ytr_acc = mnb.score(Xtr,ytr)
    yv_acc = mnb.score(Xv,yv)
    # table-ize
    output ={
            'model':'MNB',
            'params':f'alpha={a},force_alpha=True',
            'tr_acc':ytr_acc,
            'v_acc':yv_acc,
        }
    metrics = [output]
    # cycle through alpha for MNB
    for a in np.arange(.1,.6,.1):
        # naive bayes multinomial
        mnb = MultinomialNB(alpha=a)
        mnb.fit(Xtr,ytr)
        # accuracies
        ytr_acc = mnb.score(Xtr,ytr)
        yv_acc = mnb.score(Xv,yv)
        # table-ize
        output ={
                'model':'MNB',
                'params':f'alpha={a}',
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)

def cat_nb_models(Xtr,ytr,Xv,yv):
    """
    Trains and evaluates Categorical Naive Bayes models with different parameter combinations and returns the performance metrics.

    Parameters:
        Xtr (array-like): Training features.
        ytr (array-like): Training labels.
        Xv (array-like): Validation features.
        yv (array-like): Validation labels.

    Returns:
        metrics (DataFrame): A DataFrame containing performance metrics for different Categorical Naive Bayes models.
    """
    # naive bayes categorical
    cat = CategoricalNB(alpha=0,force_alpha=True)
    cat.fit(Xtr,ytr)
    # accuracies
    ytr_acc = cat.score(Xtr,ytr)
    yv_acc = cat.score(Xv,yv)
    # table-ize
    output ={
            'model':'CatNB',
            'params':f'alpha={a},force_alpha=True',
            'tr_acc':ytr_acc,
            'v_acc':yv_acc,
        }
    metrics = [output]
    # cycle through alpha for CatNB
    for a in np.arange(.1,.6,.1):
        # naive bayes categorical
        cat = CategoricalNB(alpha=a)
        cat.fit(Xtr,ytr)
        # accuracies
        ytr_acc = cat.score(Xtr,ytr)
        yv_acc = cat.score(Xv,yv)
        # table-ize
        output ={
                'model':'CatNB',
                'params':f'alpha={a}',
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)


def encode(df):
    '''Encode categorical columns'''
    # columns to encode
    cols = ['tags','product_bins']
    # encode the dummies
    dummy = pd.get_dummies(df[cols],prefix='',prefix_sep='',drop_first=True)
    # bring the dummies along
    return pd.concat([df,dummy],axis=1)


def select_kbest(X, y, k=2, scoring=chi2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    scoring: scoring type, default chi2, other category mutual_info_classif
    ---
    return: a df of the selected features from the SelectKBest process
    ---
    Format: kbest_results = function()
    '''
    kbest = SelectKBest(scoring, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    kbest_results = pd.DataFrame(
                dict(p_value=kbest.pvalues_, feature_score=kbest.scores_),
                index = X.columns)
    return kbest_results.sort_values(by=['feature_score'], ascending=False).head(k)


def multi_nb_hyperparam_search(X_train, y_train, X_val, y_val, hyperparams):
    """
    Performs a hyperparameter search for Multinomial Naive Bayes models and returns the results.

    Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        hyperparams (list): List of dictionaries, where each dictionary contains a set of hyperparameters.

    Returns:
        df_results (DataFrame): A DataFrame containing the results of the hyperparameter search.
    """
    results = []
    
    for params in hyperparams:
        model = MultinomialNB(**params)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        
        results.append({'Parameters': params, 'Train Accuracy': train_acc, 'Validation Accuracy': val_acc})
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Validation Accuracy', ascending=False).head(10)
    
    # Plotting the top 10 models
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(df_results) + 1), df_results['Train Accuracy'], label='Train Accuracy')
    plt.plot(range(1, len(df_results) + 1), df_results['Validation Accuracy'], label='Validation Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Top 10 Models - Train vs Validation Accuracy')
    plt.legend()
    plt.xticks(range(1, len(df_results) + 1))
    plt.show()
    
    return df_results


def forest_models(Xtr,ytr,Xv,yv):
    """
    The function `forest_models` trains and evaluates random forest models with different combinations
    of hyperparameters and returns a DataFrame containing the model's parameters and accuracy scores on
    the training and validation sets.
    
    :param Xtr: Xtr is the training data features. It is a matrix or dataframe containing the input
    features for training the random forest model
    :param ytr: The variable `ytr` represents the target variable (or the dependent variable) for the
    training set. It contains the actual values that we are trying to predict or classify
    :param Xv: Xv is the validation set features. It is a matrix or dataframe containing the input
    features for the validation set
    :param yv: The variable "yv" represents the target variable for the validation set. It is the true
    values of the target variable for the validation set
    :return: a pandas DataFrame containing the metrics for different random forest models. The DataFrame
    has columns for the model name, model parameters, training accuracy, and validation accuracy.
    """
    metrics = []
    # cycle through depth,leaf,class_weight for random forest
    for d,l in itertools.product(range(5,11),range(5,11)):
        # random forest
        forest = RandomForestClassifier(max_depth=d, min_samples_leaf=l,random_state=123)
        forest.fit(Xtr,ytr)
        # accuracies
        ytr_acc = forest.score(Xtr,ytr)
        yv_acc = forest.score(Xv,yv)
        # table-ize
        output ={
                'model':'Random Forest',
                'params':f"max_depth={d},min_samples_leaf={l},random_state=123",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)
