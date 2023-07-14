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
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB, CategoricalNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import nltk
import nltk.sentiment
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#feature importance
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif



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
    cv = CountVectorizer(token_pattern=r'(?u)\b\w+\b', max_features=2900)
    # fit and transform train
    Xtr_bow_cv = cv.fit_transform(Xtr.lemon)
    # transform val and test
    Xv_bow_cv = cv.transform(Xv.lemon)
    Xt_bow_cv = cv.transform(Xt.lemon)
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
    Xtr_bow_tfidf = tfidf.fit_transform(Xtr.lemon)
    # transform val and test
    Xv_bow_tfidf = tfidf.transform(Xv.lemon)
    Xt_bow_tfidf = tfidf.transform(Xt.lemon)
    # make dfs
    Xtr_tfidf = pd.DataFrame(Xtr_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xtr.index)
    Xv_tfidf = pd.DataFrame(Xv_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xv.index)
    Xt_tfidf = pd.DataFrame(Xt_bow_tfidf.todense(),columns=tfidf.get_feature_names_out(),index=Xt.index)
    return Xtr_tfidf,Xv_tfidf,Xt_tfidf



def tree_models(Xtr,ytr,Xv,yv):
    metrics = []
    # cycle through depth,leaf,class_weight for dec tree
    for d,l,cw in itertools.combinations(range(1,21),range(1,21),['balanced',None]):
        # decision tree
        tree = DecisionTreeClassifier(max_depth=d, min_samples_leaf=l,class_weight=cw,random_state=123)
        tree.fit(Xtr,ytr)
        # accuracies
        ytr_acc = tree.score(Xtr,ytr)
        yv_acc = tree.score(Xv,yv)
        # table-ize
        output ={
                'model':'Decision Tree',
                'params':f"max_depth={d},min_samples_leaf={l},class_weight={cw},random_state=123",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)

def forest_models(Xtr,ytr,Xv,yv):
    metrics = []
    # cycle through depth,leaf,class_weight for random forest
    for d,l,cw in itertools.combinations(range(1,21),range(1,21),['balanced','balanced_subsample',None]):
        # random forest
        forest = RandomForestClassifier(max_depth=d, min_samples_leaf=l,class_weight=cw,random_state=123)
        forest.fit(Xtr,ytr)
        # accuracies
        ytr_acc = forest.score(Xtr,ytr)
        yv_acc = forest.score(Xv,yv)
        # table-ize
        output ={
                'model':'Random Forest',
                'params':f"max_depth={d},min_samples_leaf={l},class_weight={cw},random_state=123",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)

def knn_models(Xtr,ytr,Xv,yv):
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
    metrics = []
    # cycle through C,class_weight for log reg
    for c,cw in itertools.combinations([.01,.1,1,10,100,1000],['balanced',None]):
        # logistic regression
        lr = LogisticRegression(C=c,class_weight=cw,random_state=123,max_iter=500)
        lr.fit(Xtr,ytr)
        # accuracies
        ytr_acc = lr.score(Xtr,ytr)
        yv_acc = lr.score(Xv,yv)
        # table-ize
        output ={
                'model':'LogReg',
                'params':f"C={c},class_weight={cw},random_state=123,max_iter=500",
                'tr_acc':ytr_acc,
                'v_acc':yv_acc,
            }
        metrics.append(output)
    return pd.DataFrame(metrics)

def comp_nb_models(Xtr,ytr,Xv,yv):
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
    # naive bayes multinomial
    mnb = MultinomialNB(alpha=0,force_alpha=True)
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
