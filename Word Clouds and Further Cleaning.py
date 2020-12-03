# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:58:40 2020

@author: Ryan Sekulic
"""

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.gridspec import GridSpec
from os import listdir
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


df = pd.read_csv("reduced_dataset.csv", index_col=0)

df = df[df['comment'] != '[removed]']

#Remove all punctuation and digits
df['comment'] = df['comment'].str.replace('[^\w\s]','')
df['comment'] = df['comment'].str.replace('[\d+]','')
#Remove all links
df['comment'] = df['comment'].str.replace("http\w+", '')

#Remove all double spaces
df['comment'] = df['comment'].str.replace('\s+', ' ')

df = df[df['comment'] != ' ']
df = df[df['comment'] != '']


# word clouds
from nltk.corpus import stopwords

stop_words = list(stopwords.words('english'))

stop_words = [i.replace("'", "") for i in stop_words]

l = len(stop_words)
for i in range(l):
    stop_words.append(stop_words[i].capitalize())

df['year'] = (pd.DatetimeIndex(df['date']).year)
df['week'] = (pd.DatetimeIndex(df['date']).week)
df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str)

year = sorted(list(set((pd.DatetimeIndex(df['date']).year.tolist()))))

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

counter = 0

for j in year:
    for z in range(1,53):
        fig = plt.figure()
        gs = GridSpec(nrows=1, ncols=2)
        for i, w in zip(['r/Canada','r/CanadaPolitics'], [0,1]):
            test = df[(df['sub'] == str(i)) & (df['year_week'] == str(j)+'-'+str(z))]
            text = " ".join(comment for comment in test.comment)
            wordcloud = WordCloud(stopwords=stop_words,
                                  width = 2500,
                                  height = 3000,
                                  background_color="black", 
                                  random_state = 9330, 
                                  collocations = True, 
                                  colormap = 'Pastel1').generate(text)
            axw = fig.add_subplot(gs[w])
            axw.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.imshow(wordcloud, interpolation='bilinear')
            axw.set_title(str(i))
        fig.suptitle(str(j) +' week '+str(z), fontsize=16)
        counter +=1
        plt.savefig('Word Clouds\\' + str(counter) + '.png')
        plt.close()
            
files = [f for f in listdir('Word Clouds\\')]

files = sorted([int(f.replace('.png','')) for f in files])

files = [str(i) +'.png' for i in files]


images = []

for i in files:
    im = Image.open('Word Clouds\\' + i)
    images.append(im)

images[0].save('GIFS\\word_clouds.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=1000, loop=1)

def lower(input_string):
    return input_string.lower()

df['comment'] = df['comment'].apply(lower)

#tokenize comments
def word_split(input_string):
    return word_tokenize(input_string, language = 'english')

df['comment'] = df['comment'].apply(word_split)

def remove_stops(list_of_strings):
    return [i for i in list_of_strings if i not in stop_words]

df['comment'] = df['comment'].apply(remove_stops)

#remove null comments less than 1 word
df = df[df['comment'].apply(len) > 0]

#stem words

stemmer = PorterStemmer()

def stem(list_of_strings):
    return [stemmer.stem(i) for i in list_of_strings]

df['comment'] = df['comment'].apply(stem)

df.to_csv("cleaned_reddit_dataset.csv")

    
    


