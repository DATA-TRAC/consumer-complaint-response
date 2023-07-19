
'''
*------------------*
|                  |
|     WRANGLE!     |
|                  |
*------------------*
'''

#------------------------------------------------------------- IMPORTS  -------------------------------------------------------------

#standard imports
import pandas as pd 
import numpy as np

#get/create files
import os
from google.oauth2 import service_account
import pandas_gbq

#splitting
from sklearn.model_selection import train_test_split

#prepare imports
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from functools import partial

#nltk sentiment
from nltk.sentiment import SentimentIntensityAnalyzer

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

print('imports loaded successfully, awaiting commands...')


'''
*------------------*
|                  |
|     FUNCTIONS    |
|                  |
*------------------*
'''

#------------------------------------------------------------- ACQUIRE -------------------------------------------------------------

# edit the below variables to your respective project, database, and table names in Google BigQuery
SQL_query = "select * from cfpb_complaints.complaint_database"
project_id = "my-ds-projects"

def check_file_exists_gbq(csv_fn, json_fn):
    '''
    check if file exists in my local directory, if not, pull from google big query db
    using service key token, return dataframe and save to a csv in the local directory
    '''
    #credentials
    credentials = service_account.Credentials.from_service_account_file(
    json_fn,)
    
    #if/else file exists in local directory
    if os.path.isfile(csv_fn):
        print('csv file found and loaded')
        return pd.read_csv(csv_fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pandas_gbq.read_gbq("select * from cfpb_complaints.complaint_database",dialect='standard',project_id="my-ds-projects",use_bqstorage_api=True,credentials=credentials)
        df.to_csv(csv_fn)
        return df
    
#------------------------------------------------------------- PREPARE -------------------------------------------------------------

def clean_data(df):
    """
    The `clean_data` function takes a DataFrame as input, performs various data cleaning operations such
    as changing column types, deriving a new column, replacing missing values, dropping rows, and
    dropping unnecessary columns, and returns the cleaned DataFrame.
    
    :param df: The parameter `df` is a pandas DataFrame that contains the data to be cleaned
    :return: a cleaned dataframe with certain columns dropped and missing values replaced.
    """
    # Change column type to datetime64[ns] for column: 'date_received'
    df = df.astype({'date_received': 'datetime64[ns]'})
    # Drop nulls based on column: 'consumer_complaint_narrative'
    df = df[df['consumer_complaint_narrative'].notna()]
    # not used anymore
    # # Derive column 'narrative' from columns: 'consumer_complaint_narrative', 'issue', 'subissue'
    # df.insert(5,'narrative',df.apply(
    #         lambda row: f'{str(row.issue)} {str(row.subissue)} {str(row.consumer_complaint_narrative)}',
    #         axis=1,),)
    # Replace missing values with "UNKNOWN" in column: 'state' and "Average Person" in column: 'tags'
    df = df.fillna({'state': "UNKNOWN",'tags': "Average Person"})
    # Drop rows with missing data in column: 'company_response_to_consumer'
    df = df.dropna(subset=['company_response_to_consumer'])
    # Filter rows based on column: 'company_response_to_consumer'
    df = df[df['company_response_to_consumer'] != "In progress"]
    # bin products into lists
    credit_report = ['Credit reporting, credit repair services, or other personal consumer reports', 'Credit reporting']
    credit_card = ['Credit card or prepaid card', 'Prepaid card', 'Credit card']
    debt_collection = ['Debt collection']
    mortgage = ['Mortgage']
    bank = ['Bank account or service', 'Checking or savings account']
    loans = ['Consumer Loan', 'Payday loan', 'Payday loan, title loan, or personal loan', 'Student loan', 'Vehicle loan or lease']
    money_service = ['Money transfer, virtual currency, or money service', 'Money transfers', 'Other financial service', 'Virtual currency']
    # make new binned product column
    df['product_bins'] = np.where(df['product'].isin(credit_report),'credit_report','')
    df['product_bins'] = np.where(df['product'].isin(credit_card),'credit_card',df['product_bins'])
    df['product_bins'] = np.where(df['product'].isin(debt_collection),'debt_collection',df['product_bins'])
    df['product_bins'] = np.where(df['product'].isin(mortgage),'mortgage',df['product_bins'])
    df['product_bins'] = np.where(df['product'].isin(bank),'bank',df['product_bins'])
    df['product_bins'] = np.where(df['product'].isin(loans),'loans',df['product_bins'])
    df['product_bins'] = np.where(df['product'].isin(money_service),'money_service',df['product_bins'])
    # drop columns not used for explore or modeling
    df = df.drop(
        columns=[
            'subproduct',
            'complaint_id',
            'consumer_disputed',
            'timely_response',
            'date_sent_to_company',
            'submitted_via',
            'company_public_response',
            'zip_code',
            'consumer_consent_provided',
            'issue',
            'subissue',
            'product'
        ]
    )
    df = df.reset_index().drop(columns='index')
    return df.rename(columns={'consumer_complaint_narrative':'narrative'})

def basic_clean(string):
    """
    The function `basic_clean` takes a string as input and performs basic cleaning operations such as
    converting the string to lowercase, removing non-alphanumeric characters, and normalizing unicode
    characters.
    
    :param string: The parameter "string" is a string that you want to perform basic cleaning on
    :return: The function `basic_clean` returns a cleaned version of the input string.
    """
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    string = re.sub(r'[^a-z0-9\'\s]', ' ', string)
    return string

def token_it_up(string):
    """
    The function `token_it_up` takes a string as input and uses the `ToktokTokenizer` from the
    `nltk.tokenize` module to tokenize the string into individual words, returning the tokenized string.
    
    :param string: The input string that you want to tokenize
    :return: a tokenized version of the input string.
    """
    tokenize = nltk.tokenize.ToktokTokenizer()
    return tokenize.tokenize(string, return_str=True)

def lemmad(string):
    """
    The function `lemmad` takes a string as input, splits it into individual words, applies
    lemmatization to each word using WordNetLemmatizer from the nltk library, and then joins the
    lemmatized words back into a string.
    
    :param string: The parameter "string" is a string of text that you want to lemmatize
    :return: The function `lemmad` returns a string where each word has been lemmatized using the
    WordNetLemmatizer from the NLTK library.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(word) for word in string.split()])

# speed things up by calling it once outside the function
corpus_words = set(nltk.corpus.words.words())

def remove_stopwords(string, extra_words=None, exclude_words=None):
    """
    The function `remove_stopwords` takes a string as input and removes common English stopwords from
    it, along with any extra words specified by the user and excluding any words specified by the user.
    
    :param string: The input string from which you want to remove stopwords. Stopwords are commonly used
    words in a language that are often considered insignificant and are typically removed from text data
    during natural language processing tasks
    :param extra_words: A list of additional words that you want to include in the stopwords list. These
    words will not be removed from the string
    :param exclude_words: A list of words that should be excluded from the stopwords list. These words
    will not be removed from the input string
    :return: a string with the stopwords removed.
    """
    if extra_words is None:
        extra_words = []
    if exclude_words is None:
        exclude_words = []

    # sls = set(stopwords.words('english'))
    # sls = set(sls) - set(exclude_words)
    # sls = sls.union(set(extra_words))
    sls = set(stopwords.words('english')) - set(exclude_words) | set(extra_words)

    # words = string.split()
    # filtered = [word for word in string.split() if word not in sls]
    string = ' '.join([word for word in string.split() if word not in sls])
    # corpus_words = set(nltk.corpus.words.words())
    return " ".join(
        w
        for w in nltk.tokenize.wordpunct_tokenize(string)
        if w.lower() in corpus_words or not w.isalpha()
    )

def clean_narrative(narrative):
    """
    The function `clean_narrative` takes a narrative as input, removes special characters, numbers, and
    apostrophes, performs basic cleaning, tokenizes the text, and returns the cleaned and tokenized
    narrative as a string.
    
    :param narrative: The `narrative` parameter is a string that represents a piece of text or a story
    :return: a cleaned and tokenized version of the input narrative.
    """
    # remove specials
    sls = ["&#9;", "12", "'"]
    cleaned = basic_clean(re.sub(r'[X{1,}\d\']', ' ', narrative))
    tokenized = token_it_up(cleaned)
    return ' '.join(word for word in tokenized.split() if word not in sls)

def prep_narrative(df):
    """
    The function `prep_narrative` takes a DataFrame as input and performs several cleaning and
    preprocessing steps on the 'narrative' column, including removing special characters, tokenizing,
    basic cleaning, removing stopwords, and lemmatizing. The function then returns the preprocessed
    DataFrame.
    
    :param df: The parameter `df` is a pandas DataFrame that contains a column named 'narrative'. This
    column contains text data that needs to be preprocessed
    :return: the prepped dataframe, which includes the original columns as well as the derived columns
    'clean' and 'lemon'.
    """
    # Derive column 'clean' from column: cleanup up 'readme_contents'
    df['clean'] = df['narrative'].apply(partial(clean_narrative))
    # Drop narrative column
    df = df.drop(columns={'narrative'})
    # Derive column 'lemmatized' from column: only real words, no stopwords 'clean'
    df['lemon'] = df['clean'].apply(partial(remove_stopwords))
    # Drop clean column
    df = df.drop(columns={'clean'})
    # lemmatize lemon
    df['lemon'] = df['lemon'].apply(partial(lemmad))
    # return prepped df
    return df

def wrangle_complaints():
    """
    The function `wrangle_complaints` checks if a prepped cache file exists, and if not, downloads a
    parquet file and caches it.
    :return: The function `wrangle_complaints()` returns a pandas DataFrame.
    """
    # check if prepped cache exists
    if os.path.isfile('cfpb_prep.parquet'):
        print('parquet file found and loaded')
        return pd.read_parquet('cfpb_prep.parquet')
    elif os.path.isfile('df_prep.parquet'):
        print('parquet file found and loaded')
        return pd.read_parquet('df_prep.parquet')
    else: 
        # get prepped df and cache it
        print('downloading file - this may take a few minutes...')
        df = pd.read_parquet('https://github.com/DATA-TRAC-DATA/large-data/raw/main/cfpb_prep.parquet')
        df.to_parquet('cfpb_prep.parquet')
        return df

def wrangle_complaints_the_long_way():
    """
    The function `wrangle_complaints` checks if a prepped cache file exists, and if not, it retrieves an
    original dataframe, cleans the data, preps the narrative using nltk, exports the prepped dataframe
    to a parquet file, and returns the prepped dataframe.
    :return: a pandas DataFrame object.
    """
    # check if prepped cache exists
    if os.path.isfile('cfpb_prep.parquet'):
        print('parquet file found and loaded')
        return pd.read_parquet('cfpb_prep.parquet')
    elif os.path.isfile('df_prep.parquet'):
        print('parquet file found and loaded')
        return pd.read_parquet('df_prep.parquet')
    else: 
        # get original df
        df = check_file_exists_gbq(csv_fn='cfpb.csv',json_fn='service_key.json')
        # clean the data up
        df = clean_data(df)
        # nltk prep the data
        df = prep_narrative(df)
        # cache prep data
        print('exporting parquet and returning df')
        df.to_parquet('cfpb_prep.parquet')
        return df

#------------------------------------------------------------- NLTK -------------------------------------------------------------

def sentiment_analysis(df):
    '''
    This function applies NLTK sentiment analysis to each narrative in the df.
    '''
    # Initialize the sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()
    # Apply the sentiment intensity analyzer to the 'lemon' column
    df['sentiment'] = df['lemon'].apply(lambda complaint: sia.polarity_scores(complaint)['compound'] if isinstance(complaint, str) else None)
    return df

#------------------------------------------------------------- SPLIT ------------------------------------------------------------

def split_data(df,variable):
    """
    This function helps divide the data into train, validate, and testing
    while stratifying on a target"
    """
    train, test = train_test_split(df,
                                    random_state=123, test_size=.20, stratify= df[variable])
    train, validate = train_test_split(train, random_state=123, test_size=.25, stratify= train[variable])
    
    #returns shapes of df's
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test