# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:32:36 2020

@author: Ryan Sekulic
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 04:51:12 2020

@author: Ryan Sekulic
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import seaborn as sns
import matplotlib.dates as mdates

df = pd.read_csv("cleaned_reddit_dataset.csv", index_col=0)

df['year'] = pd.DatetimeIndex(df['date']).year

df = df[['comment','sub','date', 'year']]

sns.set(font_scale=1.5) 
sns.set_style("white")

sample = df[['comment','sub', 'date', 'year']].sample(1500000, random_state = 9330)

def pie(data: object, groupby: str, variable: str):
    """ 
    Creates pie chart describing the desired paramter.
    
    Args:
        data: a dataframe object
        groupby: the variable by which to graph the pie chart.
        variable: the variable to count for the pie chart
    """
    pie_chart = data.groupby([groupby]).count()[[variable]]
    pie_chart.reset_index(inplace = True)
    pie, ax = plt.subplots(figsize = (12,6))
    plt.pie(x=pie_chart[variable], autopct="%.1f%%", explode=[0.05]*(data[groupby].nunique()), labels=pie_chart[groupby], pctdistance=0.5)

for i in [df, sample]:
    for j in ['sub', 'year']:
        pie(i, j, 'comment')

vectorizer = TfidfVectorizer(ngram_range = (1, 1), max_features = 1500)

X_train, X_test, y_train, y_test = train_test_split(vectorizer.fit_transform(sample['comment'].values.astype(str)).toarray(), 
                                                    sample['sub'], test_size = 0.25, random_state = 9330)

models = [
    LinearSVC(class_weight='balanced'),
    MultinomialNB(fit_prior=False),
    LogisticRegression(class_weight='balanced', random_state = 9330, solver = 'saga')
    ]

k = 5
cv_df = pd.DataFrame(index=range(k * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, X_train, y_train, scoring='balanced_accuracy', cv=k)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['Model', 'fold_idx', 'Balanced accuracy'])
sns.boxplot(x='Model', y='Balanced accuracy', data=cv_df)
sns.stripplot(x='Model', y='Balanced accuracy', data=cv_df, 
             size=8, jitter=True, edgecolor="gray", linewidth=2, palette = 'deep')
sns.despine()
plt.show()

model = LogisticRegression(class_weight = 'balanced', random_state = 9330, solver = 'saga')

param_grid = {'C': [0.01,0.1,1,10]}

grid = GridSearchCV(model, param_grid,refit=True, scoring = 'balanced_accuracy')
grid.fit(X_train,y_train)
print(grid.best_estimator_)

model = LogisticRegression(C=0.1, random_state = 9330, solver = 'saga', class_weight='balanced')
model.fit(X_train, y_train)

def results_score(X: object, y: object, model:object) -> float:
    """ Calculates the balanced accuracy score for a series of predictions versus true values
    
    Args:
        X: The input data that informs the prediction.
        y: The true values to compare predictions against.
        model: the fitted classifier
    
    Returns:
        The balanced accuracy score
    """
    y_pred = model.predict(X)
    return [balanced_accuracy_score(y, y_pred)]

print(results_score(X_train, y_train, model))
print(results_score(X_test, y_test, model))

def conf_m(X: object, y: object, title: str):
    """ Plots a confusion matrix for a series of predictions versus true values
    
    Args:
        X: The input data that informs the prediction.
        y: The true values to compare predictions against.
        title: the title to place on the confusion matrix
    """
    y_pred = model.predict(X)
    conf_mat = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=['r/Canada', 'r/CanadaPolitics'], yticklabels=['r/Canada', 'r/CanadaPolitics'], cmap = 'YlGnBu')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

conf_m(X_test, y_test, "Confusion Matrix - Test Set")

y_test = y_test.to_frame()
y_test['predictions_test'] = model.predict(X_test)
y_test['category_test'] = "Test"

y_train = y_train.to_frame()
y_train['predictions_train'] = model.predict(X_train)
y_train['category_train'] = "Train"

results = sample.merge(y_test[['predictions_test', 'category_test']], how = 'left', left_index = True, right_index = True)
results = results.merge(y_train[['predictions_train', 'category_train']], how = 'left', left_index = True, right_index = True)

results['predictions'] = results['predictions_test'].fillna(results['predictions_train'])
results['category'] = results['category_test'].fillna(results['category_train'])

results = results[['sub','predictions', 'category','date']]

results['date'] = pd.to_datetime(results['date'])

def recall_processing(subs: list, data: object) -> object:
    """ 
    Creates helper columns and then uses those columns to calculate a balanced accuracy score for each time period in the dataframe.
    Additionally converts the time period in the dataframe into months.
    
    Args:
        subs: The different subreddits in the dataframe
        data: The input dataframe.
    
    Returns:
        A modified dataframe with a balanced accuracy score for each time period. 
    """
    for i in subs:
        data[i + '_tp'] = np.where((data['sub'] == i) & (data['predictions'] == i), 1, 0)
        data[i + '_fp'] = np.where((data['sub'] != i) & (data['predictions'] == i), 1, 0)
        data[i + '_tn'] = np.where((data['sub'] != i) & (data['predictions'] != i), 1, 0)
        data[i + '_fn'] = np.where((data['sub'] == i) & (data['predictions'] != i), 1, 0)
    data['date'] = data['date'].dt.to_period('M').apply(lambda r: r.start_time)
    data = data.groupby(['date', 'category']).sum()[data.columns[4:]]
    for j in subs:
        data[j + '_recall'] = data[j +'_tp'] / (data[j +'_tp'] + data[j +'_fn'])
    data['balanced_accuracy'] = (data['r/Canada_recall'] + data['r/CanadaPolitics_recall']) / 2
    return data

results = recall_processing(['r/Canada', 'r/CanadaPolitics'], results)    

results.reset_index(inplace = True)

results['balanced_accuracy'] = results.groupby('category')['balanced_accuracy'].transform(lambda row: row.rolling(6, min_periods = 1, center = True).mean())

results['chart_dates'] = [mdates.date2num(i) for i in results['date']]

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
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.set(xlabel="Date", ylabel = y_title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:],bbox_to_anchor=(0.35, -0.15), loc=2, borderaxespad=0., ncol = 3, frameon = False)
    sns.despine()
    
line_plot("balanced_accuracy", 'Balanced accuracy', 'category', results[results['category'] == 'Test'], 'deep', 1)
line_plot("balanced_accuracy", 'Balanced accuracy', 'category', results, 'deep', 1)

   







