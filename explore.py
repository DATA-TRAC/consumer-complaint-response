# #module
# import stats_conclude as sc

#standard imports
import pandas as pd

#text
import re
import unicodedata
import nltk

#for viz
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# -----------------------------------------------------------------EXPLORE-----------------------------------------------------------------

def unique_words(word_counts):
    '''
    '''
    # setting basic style parameters for matplotlib
    plt.rc('figure', figsize=(13, 7))
    plt.style.use('seaborn-darkgrid')

    # Iterate over the columns of word_counts
    for i, col in enumerate(word_counts.columns):
        plt.subplot(3, 3, i+1)  # Adjust the subplot parameters as per your requirement

        # Sort the values in the column in descending order and select the top 5
        top_words = word_counts[col].sort_values(ascending=False).head(3)

        # Create a horizontal bar plot
        top_words.plot.barh()

        plt.xlabel('Count')
        plt.ylabel('Word')
        plt.title(f'Word Identification per Response: Sorted on {col}')

    plt.tight_layout()  # Adjust the layout to avoid overlapping subplots
    plt.show()

def basic_clean_split(string):
    """
    The function `basic_clean` takes a string as input and performs basic cleaning operations such as
    converting the string to lowercase, removing non-alphanumeric characters, and normalizing unicode
    characters.
    
    :param string: The parameter "string" is a string that you want to perform basic cleaning on
    :return: The function `basic_clean` returns a cleaned version of the input string.
    """
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode('utf-8')
    string = re.sub(r'[^a-z0-9\'\s]', ' ', string).split()
    return string

def get_words(train):
    '''
    this function extracts and counts words from a df based on different company responses.
    returns a word_count df containing the associated words for each response
    '''
    #assinging all words to proper labels
    explanation_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with explanation'].lemon))
    no_money_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with non-monetary relief'].lemon))
    money_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with monetary relief'].lemon))
    timed_out_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Untimely response'].lemon))
    closed_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed'].lemon))
    all_words = basic_clean_split(' '.join(train.lemon))
    
    #grabbing frequencies of occurences
    explanation_freq = pd.Series(explanation_words).value_counts()
    no_money_freq = pd.Series(no_money_words).value_counts()
    money_freq = pd.Series(money_words).value_counts()
    timed_out_freq = pd.Series(timed_out_words).value_counts()
    closed_freq = pd.Series(closed_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    #combine into df to see all words and languages together
    word_counts = (pd.concat([all_freq, explanation_freq, no_money_freq, money_freq, timed_out_freq, closed_freq], axis=1, sort=True)
                .set_axis(['all', 'explanation', 'no_money', 'money', 'timed_out', 'closed'], axis=1)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    print(f"Total Unique Words Found per Response: {word_counts.shape[0]}")
    print()
    
    return word_counts    

def get_words_products(train):
    '''
    this function extracts and counts words from a df based on different products.
    returns a word_count df containing the associated words for each product
    '''
    #assinging all words to proper labels
    credit_report_words = basic_clean(' '.join(train[train.product_bins == 'credit_report'].lemon))
    debt_words = basic_clean(' '.join(train[train.product_bins == 'debt_collection'].lemon))
    credit_card_words = basic_clean(' '.join(train[train.product_bins == 'credit_card'].lemon))
    mortgage_words = basic_clean(' '.join(train[train.product_bins == 'mortgage'].lemon))
    loans_words = basic_clean(' '.join(train[train.product_bins == 'loans'].lemon))
    bank_words = basic_clean(' '.join(train[train.product_bins == 'bank'].lemon))
    money_service_words = basic_clean(' '.join(train[train.product_bins == 'money_service'].lemon))
    all_words = basic_clean(' '.join(train.lemon))
    
    #grabbing frequencies of occurences
    credit_report_freq = pd.Series(credit_report_words).value_counts()
    debt_freq = pd.Series(debt_words).value_counts()
    credit_card_freq = pd.Series(credit_card_words).value_counts()
    mortgage_freq = pd.Series(mortgage_words).value_counts()
    loans_freq = pd.Series(loans_words).value_counts()
    bank_freq = pd.Series(bank_words).value_counts()
    money_service_freq = pd.Series(money_service_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()

    #combine into df to see all words and languages together
    word_counts_products = (pd.concat([all_freq, credit_report_freq, debt_freq, credit_card_freq, mortgage_freq, loans_freq, bank_freq, money_service_freq], axis=1, sort=True)
                .set_axis(['all', 'credit_report', 'debt', 'credit_card', 'mortgage', 'loans', 'bank', 'money_service'], axis=1)
                .fillna(0)
                .apply(lambda s: s.astype(int)))
    
    print(f"Total Unique Words Found per Product: {word_counts_products.shape[0]}")
    print()
    
    return word_counts_products
    
def calculate_average_letter_count(df):
    # Calculate the letter count for each row
    df['letter_count'] = df['readme'].apply(lambda x: len(x))
    # Group by language and calculate the average letter count
    grouped_data = df.groupby('language').agg('mean')
    print(grouped_data)
    # Create a bar plot
    plt.bar(grouped_data.index, grouped_data.letter_count)
    plt.xlabel('Language')
    plt.ylabel('Average Letter Count')
    plt.title('Average Letter Count by Language')
    plt.show()
    sc.compare_categorical_continuous('language', 'letter_count', df)        