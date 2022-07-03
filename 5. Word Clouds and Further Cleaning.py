# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:58:40 2020

@author: Ryan Sekulic
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


df = pd.read_csv("reduced_dataset.csv", index_col=0)

#Remove comments that were removed by moderators
df = df[df['comment'] != '[removed]']

def strip(data: object, column: str) -> object:
    """ 
    Remove all non-letter characters, all hyperlinks, and all multispaces from a series of strings.
    Subsequently remove any comments that have become empty because of the previous operations.
    
    Args:
        data: a dataframe object.
        column: a column of text within the dataframe.
    
    Returns:
        A dataframe object with the desired column cleaned.
    """
    data[column] = data[column].str.replace('[^\w\s]','')
    data[column] = data[column].str.replace('[\d+]','')
    data[column] = data[column].str.replace("http\w+", '')
    data[column] = data[column].str.replace('\s+', ' ')
    data = data[(data[column] != ' ') & (data['comment'] != '')]
    return data

df = strip(df, 'comment')

#Word clouds
stop_words = list(stopwords.words('english'))

stop_words = [i.replace("'", "") for i in stop_words]

l = len(stop_words)
for i in range(len(stop_words)):
    stop_words.append(stop_words[i].capitalize())

election_2015 = df[(df['date'] > '2015-08-03') & (df['date'] < '2015-10-21')]
election_2019 = df[(df['date'] > '2019-09-10') & (df['date'] < '2019-10-22')]

for j, z in zip([election_2015, election_2019], ['2015 Election', '2019 Election']):
    fig = plt.figure()
    gs = GridSpec(nrows=1, ncols=2)
    for i, w in zip(['r/Canada','r/CanadaPolitics'], [0,1]):
        test = j[j['sub'] == str(i)]
        text = " ".join(comment for comment in test.comment)
        wordcloud = WordCloud(stopwords=stop_words,
                              width = 2500,
                              height = 3000,
                              background_color="black", 
                              random_state = 9330, 
                              collocations = True, 
                              colormap = 'Pastel1',
                              max_words = 50).generate(text)
        axw = fig.add_subplot(gs[w])
        axw.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.imshow(wordcloud, interpolation='bilinear')
        axw.set_title(str(i))
    fig.suptitle(z, fontsize=16)

#Final pre-classification cleaning

#convert comments to lowercase
def lower(input_string: str) -> str:
    """ Convert a string to all lowercase. """
    return input_string.lower()

df['comment'] = df['comment'].apply(lower)

#tokenize comments
def word_split(input_string: str) -> list:
    """ Tokenize a string. """
    return word_tokenize(input_string, language = 'english')

df['comment'] = df['comment'].apply(word_split)

def remove_stops(list_of_strings: list) -> list:
    """ Remove all stopwords from a list of strings. """
    return [i for i in list_of_strings if i not in stop_words]

df['comment'] = df['comment'].apply(remove_stops)

#remove null comments 
df = df[df['comment'].apply(len) > 0]

#stem words
stemmer = PorterStemmer()

def stem(list_of_strings: list) -> list:
    """ Stem words in a list of strings. """
    return [stemmer.stem(i) for i in list_of_strings]

df['comment'] = df['comment'].apply(stem)

df.to_csv("cleaned_reddit_dataset.csv")
 


