# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:32:32 2020

@author: Ryan Sekulic
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib_venn import venn2
from PIL import Image
from os import listdir


df = pd.read_csv('reddit_dataset.csv', index_col = 0)

#filter based on common link
df= df[df['url'].isin(set(df[df['sub'] == 'canada']['url']).intersection(set(df[df['sub'] == 'CanadaPolitics']['url']))) == True]

#Remove any comments from the AutoModerator
df = df[df['author'] != 'AutoModerator']

#Remove memes, repeated responses, and other such spam --> Comments must be unique
df['count'] = df.groupby('comment')['comment'].transform('count')
df['count'] = np.where(df['comment'] == '[removed]', 1, df['count'])
df = df[df['count'] <= 1]

#Convert the date column into datetime
df['date'] = df['date'].apply(lambda x: datetime.datetime.fromtimestamp(x)).dt.floor('d')

#Remove comments outside of time scope
df = df[(df['date'] > '2014-12-31') & (df['date'] < '2020-10-01')]

#Correct subreddit names
df['sub'] = df['sub'].str.replace('canada', 'r/Canada')
df['sub'] = df['sub'].str.replace('CanadaPolitics', 'r/CanadaPolitics')

#reorder columns
df = df[['sub', 'date', 'post_id', 'author', 'comment', 'score', 'url']]

df.to_csv('reduced_dataset.csv')


#Generate information for rolling average weekly stats dataframe
weekly_stats = df.groupby(['date', 'sub']).count()[['comment']]
x2 = df[df['comment'] == '[removed]'].groupby(['date', 'sub']).count()[['comment']]
x3 = df.groupby(['date', 'sub']).mean()[['score']]

for i in [x2, x3]:
    weekly_stats = weekly_stats.merge(i, left_index=True, right_index=True, how='outer')
 
weekly_stats = weekly_stats.rename(columns = {'comment_x':'num_comments',
                                            'comment_y':'num_removed', 
                                            'score':'av_score'})

weekly_stats.reset_index(inplace = True)
weekly_stats = weekly_stats.sort_values('date', ascending = True)

weekly_stats['num_removed'] = weekly_stats['num_removed'].fillna(0)
weekly_stats['% removed'] = (weekly_stats['num_removed']/weekly_stats['num_comments']) * 100

weekly_stats.set_index('date', inplace = True)

for i in weekly_stats.columns[1:]:
    weekly_stats[i] = weekly_stats.groupby('sub')[i].transform(lambda row: row.rolling('7d', min_periods = 1).mean())

weekly_stats['chart_dates'] = [mdates.date2num(i) for i in weekly_stats.index]

#line plot of comments over time
sns.set(font_scale=1.5) 
sns.set_style("white")

def line_plot(y_variable: str, y_title: str, categories: str, data: object, palette: str, alpha: float):
    """ 
    Creates a line chart of the chosen variables over time
    
    Args:
        y_variable: The variable to plot on the vertical axis.
        y_title: The vertical axis title.
        categories: The variable to along which to split into different colour coded lines.
        data: The input dataframe.
        palette: The Seaborn palette to use.
        alpha: Transparency of the lines.
    """
    fig, ax = plt.subplots(figsize = (12,6))
    sns.lineplot(x='chart_dates', y=y_variable, data = data, hue = categories, palette = palette, alpha = alpha, ax = ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax.yaxis.grid(True, which='major', alpha = 0.75, linestyle = '--')
    ax.set(xlabel="Date", ylabel = y_title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:],bbox_to_anchor=(0.28, -0.15), loc=2, borderaxespad=0., ncol = 3, frameon = False)
    sns.despine()

line_plot('num_comments', 'Number of comments', 'sub', weekly_stats, 'deep', 1)

#pie chart of comment share

pie_chart = df.groupby(['sub']).count()[['comment']]
pie_chart.reset_index(inplace = True)

pie, ax = plt.subplots(figsize = (12,6))
plt.pie(x=pie_chart['comment'], autopct="%.1f%%", explode=[0.05]*2, labels=pie_chart['sub'], pctdistance=0.5)

def trend_plot(y_variable: str, y_units: str, y_title: str, categories: str, category_values: list, data: object, palette: str, alpha: float, y_decimals: int):
    """ 
    Creates a scatter plot of the chosen variables with time as the X value. 
    Layers a regression line for each variable on top of this scatter plot to help display trends. 
    
    Args:
        y_variable: The variable to plot on the vertical axis.
        y_units: The units to use for the y axis (percent or float)
        y_title: The vertical axis title.
        categories: The variable to along which to split into different colour coded lines.
        category_values: All possible category values
        data: The input dataframe.
        palette: The Seaborn palette to use.
        alpha: Transparency of the lines.
        y_decimals: The number of decimal points to use on the y axis.
    """
    fig, ax = plt.subplots(figsize = (12,6))
    sns.scatterplot(x='chart_dates', y=y_variable, data = data, hue = categories, palette = palette, alpha = alpha, ax = ax)
    sns.regplot(x='chart_dates', y=y_variable, data = data[data[categories] == category_values[0]], color = sns.color_palette(palette)[0], marker = '', ax = ax)
    sns.regplot(x='chart_dates', y=y_variable, data = data[data[categories] == category_values[1]], color = sns.color_palette(palette)[1],  marker = '', ax = ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    if y_units == 'percent':
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals = y_decimals))
    ax.yaxis.grid(True, which='major', alpha = 0.75, linestyle = '--')
    ax.set(xlabel="Date", ylabel = y_title)
    handles, labels = ax.get_legend_handles_labels()
    labels = category_values
    ax.legend(handles=handles[1:], labels=labels[:],bbox_to_anchor=(0.28, -0.15), loc=2, borderaxespad=0., ncol = 3, frameon = False)
    sns.despine()
   
trend_plot("av_score", 'integer', 'Average net upvote score', 'sub', ['r/Canada', 'r/CanadaPolitics'], weekly_stats, 'deep', 0.2, 0)
trend_plot("% removed", 'percent', 'Percentage of comments moderator removed', 'sub', ['r/Canada', 'r/CanadaPolitics'], weekly_stats, 'deep', 0.2, 0)

#Venn Diagrams

vdata = df.groupby(['date','author','sub']).count()[['comment']].reset_index()[['date','author','sub']]

vdata = pd.concat([vdata, pd.get_dummies(vdata['sub'])], axis = 1).drop(columns = ['sub'])

vdata = vdata[vdata['author'] != 'nan']

vdata = vdata.groupby(['date', 'author']).sum()[['r/CanadaPolitics','r/Canada']]

vdata['cp_ca'] = vdata['r/CanadaPolitics'] * vdata['r/Canada']

vdata = vdata.groupby(['date']).sum()

for i in vdata.columns[1:]:
    vdata[i] = vdata[i].transform(lambda row: round(row.rolling('7d', min_periods = 1).mean()))

year = pd.DatetimeIndex(vdata.index).year.tolist()
month = pd.DatetimeIndex(vdata.index).month.tolist()
day = pd.DatetimeIndex(vdata.index).day.tolist()

def venn(year: int, month: int, day: int, data: object, filenumber: float):
    """ 
    Creates a venn diagram of users between subreddits. 
    Subsequently saves that venn diagram as a file, for later use in producing an animation. 
    
    Args:
        year: The year to filter the input data on.
        month: The month to filter the input data on.
        day: The day to filter the input data on.
        data: The input data.
        filenumber: The number to name the file as when saving.
    """
    v1 = data[(pd.DatetimeIndex(vdata.index).year == year) & (pd.DatetimeIndex(vdata.index).month == month) & (pd.DatetimeIndex(vdata.index).day == day)]
    values = tuple(v1[['r/CanadaPolitics','r/Canada','cp_ca']].astype(int).values[0])
    labels = ['r/CanadaPolitics', 'r/Canada']
    plt.figure()
    ax = plt.gca()
    v = venn2(subsets = values, set_labels = None, alpha = 1, ax = ax, set_colors = (sns.color_palette('deep')[1], sns.color_palette('deep')[0]))
    h, l = [],[]
    for i, j in zip(['10', '01'], labels):
        h.append(v.get_patch_by_id(i))   
        l.append(j)
    ax.legend(handles = h, labels = l ,bbox_to_anchor=(0.01, 0.05), loc=2, borderaxespad=0., ncol = 3, frameon = False)       
    plt.title(str(year) + '-' + str(month) + '-' + str(day))       
    plt.savefig('Venn Diagrams\\' + str(filenumber) + '.png')
    plt.close()

for i, j, w, z in zip(year, month, day, range(len(year))):
    venn(i, j, w, vdata, z)  

vdata['c_perc'] = (vdata['cp_ca']/(vdata['r/Canada']+vdata['cp_ca']))* 100 
vdata['cp_perc'] = (vdata['cp_ca']/(vdata['r/CanadaPolitics']+vdata['cp_ca'])) * 100
vdata[['c_perc', 'cp_perc']] = vdata[['c_perc', 'cp_perc']].rolling(7).mean()
vdata['chart_dates'] = [mdates.date2num(i) for i in vdata.index]

melt = pd.melt(vdata, id_vars = ['chart_dates'], value_vars = ['r/Canada', 'r/CanadaPolitics'])

line_plot('value', 'Userbase', 'variable', melt, 'deep', 1)

melt = pd.melt(vdata, id_vars = ['chart_dates'], value_vars = ['c_perc', 'cp_perc'])
melt = melt.dropna()

trend_plot("value", 'percent', 'Percentage crossover', 'variable', ['c_perc', 'cp_perc'], melt, 'deep', 0.2, 1, ['r/Canada', 'r/CanadaPolitics'])

files = [f for f in listdir('Venn Diagrams\\')]

files = sorted([int(f.replace('.png','')) for f in files])

files = [str(i) +'.png' for i in files]

images = []

for i in files:
    im = Image.open('Venn Diagrams\\' + i)
    images.append(im)

images[0].save('GIFS\\venns.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=100, loop=1)

    
