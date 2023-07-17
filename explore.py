'''
*------------------*
|                  |
|     EXPLORE!     |
|                  |
*------------------*
'''



#------------------------------------------------------------- IMPORTS  -------------------------------------------------------------
#standard imports
import pandas as pd
import numpy as np
from scipy import stats

#text
import re
import unicodedata
import nltk

#for viz
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle as w

'''
*------------------*
|                  |
|     FUNCTIONS    |
|                  |
*------------------*
'''

# -----------------------------------------------------------------EXPLORE-----------------------------------------------------------------

def unique_words(word_counts):
    """
    The function `unique_words` takes a DataFrame `word_counts` as input and creates a subplot of
    horizontal bar plots, each showing the top 3 words with the highest count for a specific column in
    the DataFrame.
    
    :param word_counts: The parameter `word_counts` is expected to be a DataFrame containing word counts
    for different responses. Each column of the DataFrame represents a different response, and each row
    represents a different word. The values in the DataFrame represent the count of each word in each
    response
    """
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
    #assigning all words to proper labels
    explanation_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with explanation'].lemon.astype(str)))
    no_money_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with non-monetary relief'].lemon.astype(str)))
    money_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed with monetary relief'].lemon.astype(str)))
    timed_out_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Untimely response'].lemon.astype(str)))
    closed_words = basic_clean_split(' '.join(train[train.company_response_to_consumer == 'Closed'].lemon.astype(str)))
    all_words = basic_clean_split(' '.join(train.lemon))
    
    #grabbing frequencies of occurrences
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
    #assigning all words to proper labels
    credit_report_words = basic_clean_split(' '.join(train[train.product_bins == 'credit_report'].lemon))
    debt_words = basic_clean_split(' '.join(train[train.product_bins == 'debt_collection'].lemon))
    credit_card_words = basic_clean_split(' '.join(train[train.product_bins == 'credit_card'].lemon))
    mortgage_words = basic_clean_split(' '.join(train[train.product_bins == 'mortgage'].lemon))
    loans_words = basic_clean_split(' '.join(train[train.product_bins == 'loans'].lemon))
    bank_words = basic_clean_split(' '.join(train[train.product_bins == 'bank'].lemon))
    money_service_words = basic_clean_split(' '.join(train[train.product_bins == 'money_service'].lemon))
    all_words = basic_clean_split(' '.join(train.lemon))
    
    #grabbing frequencies of occurrences
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

def analyze_sentiment(train,alpha=0.05,truncate=False):
    """Analyzes sentiment and company response to consumer across product bins.
    This function answers the question: Do narratives with a neutral or positive sentiment
    analysis relating to bank account products lead to a response of closed with monetary relief?"""

    # Running sentiment analysis and adding compound scores into the sentiment df. 
    sentiment_df=w.sentiment_analysis(train)
    # Set the figure size
    plt.figure(figsize=(10, 6))

    # Customize the plot style
    sns.set(style="whitegrid")

    # Create the bar plot
    sns.barplot(data=sentiment_df, x='product_bins', y='sentiment', hue='company_response_to_consumer', ci=None, color='purple')

    # Set the labels and title
    plt.xlabel('Product Bins')
    plt.ylabel('Sentiment')
    plt.title('Company Response to Consumer and Sentiment Analysis across Product Bins')

    # Adjust the legend position
    plt.legend(loc='best')

    # Show the plot
    plt.show()

    # Create example data for Levene test
    group1 = np.random.normal(loc=10, scale=2, size=100)
    group2 = np.random.normal(loc=12, scale=2, size=100)

    # Calculate the theoretical means for each group
    theoretical_mean_group1 = np.mean(group1)
    theoretical_mean_group2 = np.mean(group2)

    # Perform Levene test for variance comparison
    tstat, pvalue = stats.levene(group1, group2)

    print("Running Levene Test...")
    if pvalue > alpha:
        print(f'p-value: {pvalue:.10f} > {alpha}?')
        print()
        print("Variance is true, proceed with ANOVA test...")
    else:
        print("p-value:", pvalue)
        print()
        print("Variance is not true. Consider alternative tests for comparing groups.")
    print()
    # Get unique categories of product_bins
    unique_bins = sentiment_df['product_bins'].unique()

    # Perform ANOVA test for each category of product_bins
    for bin_category in unique_bins:
        # Create a subset of the data for the specific product_bins category
        subset = sentiment_df[sentiment_df['product_bins'] == bin_category]

        # Perform one-way ANOVA for the subset
        result = stats.f_oneway(*[subset[subset['company_response_to_consumer'] == response]['sentiment']
                                    for response in subset['company_response_to_consumer'].unique()])

        # Print the ANOVA test result for the subset
        print("Product Bins:", bin_category)
        print("ANOVA p-value:", result.pvalue)

        if result.pvalue < alpha:
            print("There is a significant effect of sentiment on company response to the consumer.")
        else:
            print("There is no significant effect of sentiment on company response to the consumer.")

        print()  # Print an empty line between each category's results

def analyze_message_length(sentiment_df, alpha=0.05):
    """
    Analyzes the relationship between message length and company response to the consumer.
    This function answers the question: Does narrative length relate to company response?
    """

    # Create the scatter plot
    plt.scatter(sentiment_df['message_length'], sentiment_df['company_response_to_consumer'], cmap='Set1')

    # Set the labels and title
    plt.xlabel('Message Length')
    plt.ylabel('Company Response to Consumer')
    plt.title('Relationship between Message Length and Company Response to Consumer')

    # Show the plot
    plt.show()

    # Perform ANOVA test
    # The code then uses a list comprehension to iterate over each unique category.
    result = stats.f_oneway(*[sentiment_df[sentiment_df['company_response_to_consumer'] == response]['message_length']
                                for response in sentiment_df['company_response_to_consumer'].unique()])

    p_value = result.pvalue

    print("ANOVA p-value:", p_value)
    if p_value < alpha:
        print("The p-value is less than alpha. There is a significant relationship between message length and company response to the consumer.")
    else:
        print("The p-value is greater than or equal to alpha. There is no significant relationship between message length and company response to the consumer.")  

def analyze_word_count(sentiment_df, alpha=0.05):
    """
    Analyzes the relationship between word count and company response to the consumer.
    This function answers the question: Does narrative word count relate to company response?
    """

    # Create the scatter plot
    plt.scatter(sentiment_df['message_length'], sentiment_df['company_response_to_consumer'], cmap='Set1')

    # Set the labels and title
    plt.xlabel('Message Length')
    plt.ylabel('Company Response to Consumer')
    plt.title('Relationship between Message Length and Company Response to Consumer')

    # Show the plot
    plt.show()

    # Perform ANOVA test
    # The code then uses a list comprehension to iterate over each unique category.
    result = stats.f_oneway(*[sentiment_df[sentiment_df['company_response_to_consumer'] == response]['message_length']
                                for response in sentiment_df['company_response_to_consumer'].unique()])

    p_value = result.pvalue

    print("ANOVA p-value:", p_value)
    if p_value < alpha:
        print("The p-value is less than alpha. There is a significant relationship between message length and company response to the consumer.")
    else:
        print("The p-value is greater than or equal to alpha. There is no significant relationship between message length and company response to the consumer.")
    # Create the scatter plot
    plt.scatter(sentiment_df['word_count'], sentiment_df['company_response_to_consumer'], cmap='Reds')

    # Set the labels and title
    plt.xlabel('Word Count')
    plt.ylabel('Company Response to Consumer')
    plt.title('Relationship between Word Count and Company Response to Consumer')

    # Show the plot
    plt.show()

    # Perform ANOVA test
    # The code uses a list comprehension to iterate over each unique category.
    result = stats.f_oneway(*[sentiment_df[sentiment_df['company_response_to_consumer'] == response]['word_count']
                                for response in sentiment_df['company_response_to_consumer'].unique()])

    p_value = result.pvalue

    print("ANOVA p-value:", p_value)
    if p_value < alpha:
        print("The p-value is less than alpha. There is a significant relationship between word count and company response to the consumer.")
    else:
        print("The p-value is greater than or equal to alpha. There is no significant relationship between word count and company response to the consumer.")


def monetary_product(train):
    """
    The function `monetary_product` creates a bar chart showing the proportions of monetary relief for
    different product types based on a given dataset.
    
    :param train: The `train` parameter is a DataFrame that contains the training data for the model. It
    should have columns named 'product_bins' and 'company_response_to_consumer'. The 'product_bins'
    column represents the different types of products, and the 'company_response_to_consumer' column
    represents the response of the
    """
    # make crosstab of product and responses and normalize to get product proportions
    cross = pd.crosstab(train['product_bins'],train['company_response_to_consumer'],normalize='index')
    # plot monetary relief products
    cross['Closed with monetary relief'].sort_values(
        ).plot(kind='barh', 
                title='Proportions of Monetary Relief', 
                xlabel='Proportion of Complaints for the Product', 
                ylabel='Product Type');


def get_word_counts(train):
    """
    Question 1 - Lugo 
    Calculates the word counts for each response type in the given training data.

    Parameters:
        train (DataFrame): The training data containing the 'lemon' and 'company_response_to_consumer' columns.

    Returns:
        word_counts (DataFrame): A DataFrame that shows the frequency of each word for each response type.
        df_with_words (DataFrame): The original DataFrame merged with the word DataFrame.
        word_counts_ones (DataFrame): A filtered version of word_counts, excluding columns with all zero values.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    
    # Initialize a CountVectorizer (or TfidfVectorizer)
    vectorizer = CountVectorizer(max_features=20000,lowercase=False)

    # Fit the vectorizer to the 'lemon' column and transform the column into a matrix
    word_matrix = vectorizer.fit_transform(train['lemon'].astype(str))

    # Convert the sparse matrix to a DataFrame
    words_df = pd.DataFrame.sparse.from_spmatrix(word_matrix, columns=vectorizer.get_feature_names_out())

    # merge the word DataFrame with the 'company_response_to_consumer' column
    df_with_words = words_df.merge(train['company_response_to_consumer'], left_index=True, right_index=True)

    # For each response type, count the frequency of each word
    word_counts = df_with_words.groupby('company_response_to_consumer').sum()

    # Filter out columns (axis=1) where all values are zero
    word_counts_ones = word_counts.loc[:, word_counts.any(axis=0)]
    
    return word_counts, df_with_words,word_counts_ones
def top_15_words(word_counts_ones):
    """
    continuation of Q1
    Retrieves the top 15 most frequently occurring words for each response type from the given word counts DataFrame.

    Parameters:
        word_counts_ones (DataFrame): The word counts DataFrame, filtered to exclude columns with all zero values.

    Returns:
        top_words_df (DataFrame): A DataFrame containing the top 15 words for each response type, indexed by the response types.
    """
    # Define the responses
    responses = ['Closed with explanation', 'Closed', 'Closed with monetary relief', 'Closed with non-monetary relief', 'Untimely response']

    # Initialize an empty DataFrame to store the results
    top_words_df = pd.DataFrame()

    # Loop over the responses
    for response in responses:
        # Get the 10 words that appear most frequently in narratives associated with the current response
        top_words = word_counts_ones.loc[response].nlargest(15)

        # Convert the Series to a DataFrame and transpose it
        top_words_df_temp = pd.DataFrame(top_words).transpose()

        # Append the temporary DataFrame to the main DataFrame
        top_words_df = pd.concat([top_words_df, top_words_df_temp])

    # Set the index of the DataFrame to the responses
    top_words_df.index = responses
    return top_words_df
def frequent_words_plot(df_with_words,word_counts_ones):
    """
    Continuation of Q1
    Creates a bar plot to visualize the top 10 most frequently occurring words for each response type.

    Parameters:
        df_with_words (DataFrame): The DataFrame merged with the word DataFrame.
        word_counts_ones (DataFrame): The filtered word counts DataFrame.

    Returns:
        None (displays the plot)
    """
    # Get the unique response types
    response_types = df_with_words['company_response_to_consumer'].unique()

    # For each response type
    for response in response_types:
        # Get the top 10 words
        top_words = word_counts_ones.loc[response].nlargest(10)

        # Create a bar plot
        plt.figure(figsize=(10, 5))
        plt.bar(top_words.index, top_words.values)
        plt.title(f'Top 10 words for "{response}" response')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
    return plt.show()
