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

    #viz
    word_counts.sort_values('all', ascending=False)[1:21][['python', 'javascript', 'html', 'shell', 'java', 'go']].head(20).plot.barh()
    plt.xlabel('Count')
    plt.ylabel('Word')
    plt.title('Word Identification per Language')
    plt.show()

def show_counts_and_ratios(df, column):
    """
    Takes in a dataframe and a string of a single column
    Returns a dataframe with absolute value counts and percentage value counts
    """
    labels = pd.concat([df[column].value_counts(),
                    df[column].value_counts(normalize=True)], axis=1).round(2)
    labels.columns = ['n', 'percent']
    labels
    return labels

def clean(text):
    '''
    a simple function to cleanup text data. returns a list of lemmatized words after cleaning.
    '''
    # Normalize text by removing diacritics, encoding to ASCII, decoding to UTF-8, and converting to lowercase
    text = (unicodedata.normalize('NFKD', text)
             .encode('ascii', 'ignore')
             .decode('utf-8', 'ignore') #most frequently used for base text creation - works great with SQL
             .lower())
    
    # Remove punctuation, split text into words
    words = re.sub(r'[^\w\s]', '', text).split()
    
    # Initialize WordNet lemmatizer
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Combine standard English stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    
    # Lemmatize words and remove stopwords
    cleaned_words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    
    return cleaned_words

def get_words(train):
    '''
    this function extracts and counts words from a df based on different company responses.
    returns a word_count df containing the associated words for each response
    '''
    #assinging all words to proper labels
    explanation_words = (' '.join(train[train.company_response_to_consumer == 'Closed with explanation'].lemon))
    no_money_words = (' '.join(train[train.company_response_to_consumer == 'Closed with non-monetary relief'].lemon))
    money_words = (' '.join(train[train.company_response_to_consumer == 'Closed with monetary relief'].lemon))
    timed_out_words = (' '.join(train[train.company_response_to_consumer == 'Untimely response'].lemon))
    closed_words = (' '.join(train[train.company_response_to_consumer == 'Closed'].lemon))
    all_words = (' '.join(train.lemon))
    
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
    
    print(f"Total Unique Words Found per Response:{word_counts.shape[0]}")
    print()
    
    return word_counts
    

def plot_unique_words_and_compare(df):
    '''
    this function uses the nested function to perform a stats test 
    '''
    def plot_unique_words_per_language(df):
        '''
        plots the count of unique words used per language (excluding 'all' and 'other') and compares them.
        '''
        word_counts = df.groupby('language')['text'].transform(lambda x: len(set(x.str.split().sum())))
        word_counts = word_counts.drop(columns={'all', 'other'})
        word_counts.nunique().plot.barh()
        plt.title('Unique Words Used Per Language (excluding other)')
        plt.xlabel('Count')
        plt.ylabel('Language')
        plt.show()

    plot_unique_words_per_language(df)
    sc.compare_categorical_continuous(df['language'], df['count'], df)



def plot_language_distribution(df):
    '''
    generates word clouds for the top 20 most frequently used words in each language.
    '''
    plt.figure(figsize=(10, 6))
    sns.countplot(y='language', data=df, order=df['language'].value_counts().index, color='skyblue')
    plt.title('Distribution of Languages')
    plt.xlabel('Count')
    plt.ylabel('Language')
    plt.show()


def generate_language_wordclouds(df):
    # Group the DataFrame by the 'language' column and join all the readme files for each language into a single string
    grouped = df.groupby('language')['readme'].apply(' '.join).reset_index()

    # Initialize a CountVectorizer
    cv = CountVectorizer(stop_words='english')

    # Ccreate a word cloud of the top 20 most frequently used words, for each language in df
    for i, row in grouped.iterrows():
        # Count the frequency of each word in the readme files for the current language
        word_count = cv.fit_transform([row['readme']])
        words = cv.get_feature_names_out()
        word_freq = word_count.toarray().sum(axis=0)
        word_freq_dict = dict(zip(words, word_freq))

        # Get the top 20 most frequently used words and their frequencies
        top_words = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        top_words_dict = dict(top_words)

        # Create a word cloud
        wc = WordCloud(width=800, height=400, max_words=20, background_color='white').generate_from_frequencies(top_words_dict)

        # Display the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Top 20 Words for {row['language']}")
        plt.show()

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