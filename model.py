# IMPORTS

'''
*------------------*
|                  |
|     MODELING     |
|                  |
*------------------*
'''

##### IMPORTS #####

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import nltk
import nltk.sentiment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#feature importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



"""
*------------------*
|                  |
|    FUNCTIONS     |
|                  |
*------------------*
"""


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
    cv = CountVectorizer(ngram_range=(1,3),token_pattern=r'(?u)\b\w+\b')
    # fit and transform train
    Xtr_bow_cv = cv.fit_transform(Xtr.lemmatized)
    # transform val and test
    Xv_bow_cv = cv.transform(Xv.lemmatized)
    Xt_bow_cv = cv.transform(Xt.lemmatized)
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
    tfidf = TfidfVectorizer(ngram_range=(1,3),token_pattern=r'(?u)\b\w+\b')
    # fit and transform train
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemmatized)
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemmatized)
    Xt_bow_tfidf = tfidf.transform(Xt.lemmatized)
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf




def class_models(Xtr,ytr,Xv,yv):
    """
    The function `class_models` trains and evaluates different classification models (Logistic
    Regression, Complement Naive Bayes, and Multinomial Naive Bayes) on the given training and
    validation data, and returns a dataframe containing the model names, parameters, training accuracy,
    and validation accuracy for each model.
    
    :param Xtr: The training data features (X) for the classification models
    :param ytr: The training labels for the classification model
    :param Xv: Xv is the feature matrix for the validation set. It contains the input features for each
    instance in the validation set
    :param yv: The parameter `yv` represents the target variable for the validation set. It is the true
    labels or classes for the validation set data
    :return: a pandas DataFrame containing the metrics for different models and their corresponding
    parameters. The metrics include the model name, parameters, training accuracy, and validation
    accuracy.
    """
    # baseline as mean
    pred_mean = ytr.value_counts(normalize=True)[0]
    output = {
            'model':'bl',
            'params':'None',
            'tr_acc':pred_mean,
            'v_acc':'?',
        }
    metrics = [output]
    # cycle through C,class_weight for log reg
    for c in [.01,.1,1,10,100,1000]:
        # logistic regression
        lr = LogisticRegression(C=c,class_weight='balanced',random_state=42,max_iter=500)
        lr.fit(Xtr,ytr)
        # accuracies
        ytr_acc = lr.score(Xtr,ytr)
        yv_acc = lr.score(Xv,yv)
        # table-ize
        output ={
                'model':'LogReg',
                'params':f"C={c},class_weight='balanced',max_iter=500",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    # cycle through alpha for CNB
    for a in np.arange(.1,.6,.1):
        # naive bayes complement
        cnb = nb.ComplementNB(alpha=a)
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
    metrics_df = pd.DataFrame(metrics)
    # cycle through alpha for MNB
    for a in np.arange(.1,.6,.1):
        # naive bayes multinomial
        mnb = nb.MultinomialNB(alpha=a)
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
    metrics_df = pd.DataFrame(metrics)
    return metrics_df





def select_kbest(X, y, k=2):
    '''
    will take in two pandas objects:
    X: a dataframe representing numerical independent features
    y: a pandas Series representing a target variable
    k: a keyword argument defaulted to 2 for the number of ideal features we elect to select
    ---
    return: a df of the selected features from the SelectKBest process
    ---
    Format: kbest_results = function()
    '''
    kbest = SelectKBest(chi2, k=k)
    kbest.fit(X, y)
    mask = kbest.get_support()
    kbest_results = pd.DataFrame(
                dict(p_value=kbest.pvalues_, feature_score=kbest.scores_),
                index = X.columns)
    return kbest_results.sort_values(by=['feature_score'], ascending=False).head(k)




def multi_nb_hyperparam_search(X_train, y_train, X_val, y_val, hyperparams):
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
