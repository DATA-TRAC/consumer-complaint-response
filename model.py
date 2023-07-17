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


#------------------------------------------------------------- MODELING-------------------------------------------------------------




def process_data_modeling():
    train = pd.read_parquet('train.parquet')
    validate = pd.read_parquet('validate.parquet')
    test = pd.read_parquet('test.parquet')
    
    sm_train1 = int(round(len(train[train.company_response_to_consumer=='Closed with explanation'])*.2,0))
    sm_train2 = int(round(len(train[train.company_response_to_consumer=='Closed with non-monetary relief'])*.2,0))
    sm_train3 = int(round(len(train[train.company_response_to_consumer=='Closed with monetary relief'])*.2,0))
    sm_train4 = int(round(len(train[train.company_response_to_consumer=='Untimely response'])*.2,0))
    sm_train5 = int(round(len(train[train.company_response_to_consumer=='Closed'])*.2,0))
    sm_val1 = int(round(len(validate[validate.company_response_to_consumer=='Closed with explanation'])*.2,0))
    sm_val2 = int(round(len(validate[validate.company_response_to_consumer=='Closed with non-monetary relief'])*.2,0))
    sm_val3 = int(round(len(validate[validate.company_response_to_consumer=='Closed with monetary relief'])*.2,0))
    sm_val4 = int(round(len(validate[validate.company_response_to_consumer=='Untimely response'])*.2,0))
    sm_val5 = int(round(len(validate[validate.company_response_to_consumer=='Closed'])*.2,0))
    sm_test1 = int(round(len(test[test.company_response_to_consumer=='Closed with explanation'])*.2,0))
    sm_test2 = int(round(len(test[test.company_response_to_consumer=='Closed with non-monetary relief'])*.2,0))
    sm_test3 = int(round(len(test[test.company_response_to_consumer=='Closed with monetary relief'])*.2,0))
    sm_test4 = int(round(len(test[test.company_response_to_consumer=='Untimely response'])*.2,0))
    sm_test5 = int(round(len(test[test.company_response_to_consumer=='Closed'])*.2,0))
    
    small_train1 = train[train.company_response_to_consumer=='Closed with explanation'].sample(sm_train1, random_state=123)
    small_train2 = train[train.company_response_to_consumer=='Closed with non-monetary relief'].sample(sm_train2, random_state=123)
    small_train3 = train[train.company_response_to_consumer=='Closed with monetary relief'].sample(sm_train3, random_state=123)
    small_train4 = train[train.company_response_to_consumer=='Untimely response'].sample(sm_train4, random_state=123)
    small_train5 = train[train.company_response_to_consumer=='Closed'].sample(sm_train5, random_state=123)
    small_val1 = validate[validate.company_response_to_consumer=='Closed with explanation'].sample(sm_val1, random_state=123)
    small_val2 = validate[validate.company_response_to_consumer=='Closed with non-monetary relief'].sample(sm_val2, random_state=123)
    small_val3 = validate[validate.company_response_to_consumer=='Closed with monetary relief'].sample(sm_val3, random_state=123)
    small_val4 = validate[validate.company_response_to_consumer=='Untimely response'].sample(sm_val4, random_state=123)
    small_val5 = validate[validate.company_response_to_consumer=='Closed'].sample(sm_val5, random_state=123)
    small_test1 = test[test.company_response_to_consumer=='Closed with explanation'].sample(sm_test1, random_state=123)
    small_test2 = test[test.company_response_to_consumer=='Closed with non-monetary relief'].sample(sm_test2, random_state=123)
    small_test3 = test[test.company_response_to_consumer=='Closed with monetary relief'].sample(sm_test3, random_state=123)
    small_test4 = test[test.company_response_to_consumer=='Untimely response'].sample(sm_test4, random_state=123)
    small_test5 = test[test.company_response_to_consumer=='Closed'].sample(sm_test5, random_state=123)

    small_train = pd.concat([small_train1, small_train2, small_train3, small_train4, small_train5])
    small_val = pd.concat([small_val1, small_val2, small_val3, small_val4, small_val5])
    small_test = pd.concat([small_test1, small_test2, small_test3, small_test4, small_test5])

    X_train = m.encode(small_train)
    X_train = X_train.drop(columns=['date_received','company_response_to_consumer','clean','state','company_name','tags','product_bins'])
    y_train = small_train['company_response_to_consumer']
    X_val = m.encode(small_val)
    X_val = X_val.drop(columns=['date_received','company_response_to_consumer','clean','state','company_name','tags','product_bins'])
    y_val = small_val['company_response_to_consumer']
    X_test = m.encode(small_test)
    X_test = X_test.drop(columns=['date_received','company_response_to_consumer','clean','state','company_name','tags','product_bins'])
    y_test = small_test['company_response_to_consumer']

    return X_train, y_train, X_val, y_val, X_test, y_test



def process_vect_merge():
    X_train_cv, X_val_cv, X_test_cv = make_cv(X_train[['lemon']], X_val[['lemon']], X_test[['lemon']])
    X_train_tf, X_val_tf, X_test_tf = make_tfidf(X_train[['lemon']], X_val[['lemon']], X_test[['lemon']])
    
    encoded_train = X_train.iloc[:, 1:]
    encoded_val = X_val.iloc[:, 1:]
    encoded_test = X_test.iloc[:, 1:]
    
    X_train_cve = encoded_train.merge(X_train_cv, left_index=True, right_index=True)
    X_val_cve = encoded_val.merge(X_val_cv, left_index=True, right_index=True)
    X_test_cve = encoded_test.merge(X_test_cv, left_index=True, right_index=True)
    
    X_train_tfe = encoded_train.merge(X_train_tf, left_index=True, right_index=True)
    X_val_tfe = encoded_val.merge(X_val_tf, left_index=True, right_index=True)
    X_test_tfe = encoded_test.merge(X_test_tf, left_index=True, right_index=True)
    
    # Visualization of train data
    return X_train_cve, X_train_tfe

# Calling the function
X_train_cve_head, X_train_tfe_head = process_data()
print("X_train_cve_head:")
print(X_train_cve_head)
print("X_train_tfe_head:")
print(X_train_tfe_head)




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
