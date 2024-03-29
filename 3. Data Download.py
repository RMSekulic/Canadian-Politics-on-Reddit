# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:58:21 2020

@author: Ryan Sekulic
"""

import pandas as pd
import requests
import json
from datetime import datetime
import praw

#Defining the Pushshift query to select the time frame to pull from, and the post data to collect

def PushShift_Query(after: int, before: int, sub: str) -> list:
    """ 
    Queries the Pushshift API for a list of posts to extract data from.
    
    Args:
        after: The date after which to search for posts
        before: The date before which to search for posts.
        sub: The subreddit to search for posts on. 
    
    Returns:
        A list of dictionaries containing post information.
    """
    link = 'https://api.pushshift.io/reddit/search/submission/?after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)+'&size=100'
    r = requests.get(link)
    print(link)
    print('Start Date ' + str(datetime.fromtimestamp(after)))
    try:
        return json.loads(r.text)['data']
    except ValueError:
        print("JSON Decode error - trying again")
        return "try again"

def collectPostData(post: dict) -> list:
    """ 
    Collects a specific list of values from a dictionary
    
    Args:
        post: The dictionary to extract values from.
    
    Returns:
        A list of values from the dictionary.
    """
    post_id = post['id']
    title = post['title']
    url = post['url']
    date = datetime.fromtimestamp(post['created_utc'])
    author = post['author']
    score = post['score']
    permalink = post['permalink']
    sub = post['subreddit']
    comments = post['num_comments']
    return [post_id, title, url, date, author, comments, score, permalink, sub]

subs = ["CanadaPolitics", "canada"]
poststats = {}

#Scraping the post level data
for i in subs:
    after = 1420070400
    before = 1601424000
    pull = PushShift_Query(after, before, i)
    while len(pull)>0:
        print('End Date ' + str(datetime.fromtimestamp(pull[-1]['created_utc'])))
        for post in pull:
            poststats[collectPostData(post)[0]] = collectPostData(post)[1:]
        after = pull[-1]['created_utc']
        pull = PushShift_Query(after, before, i)
        while pull == "try again":
            pull = PushShift_Query(after, before, i)

#sanity check
print(len(poststats.keys()) == len(set(poststats.keys())))

postcount = len(poststats)

save = pd.DataFrame.from_dict(poststats, orient = 'index')
save.to_csv('posts.csv')

#Scraping the comments from the previously collected posts

reddit = praw.Reddit(client_id='REDACTED', 
                     client_secret='REDACTED', 
                     user_agent='REDACTED')
  
remaining = [key for key in poststats]

comments = []
empties = []
unable = []
counter = 0

download = [key for key in remaining]
for post_id in download:
    counter +=1
    if counter % (round(postcount/20)) == 0: 
        print("{:.2%}".format(counter/postcount))
    post = reddit.submission(id=post_id)
    try:
        post.comments.replace_more(limit=None)
        if len(post.comments.list()) == 0: 
            empties.append(post_id)
            remaining.remove(post_id)
            pass
        else: 
            for comment in post.comments.list():
                comments.append([comment.body, comment.score, post_id, poststats[post_id][0], comment.author, poststats[post_id][7], comment.created_utc, poststats[post_id][1]])
            remaining.remove(post_id)
    except: 
        unable.append(post_id)
        remaining.remove(post_id)      
     
remaining_df = pd.DataFrame(remaining, columns = ["post_id"]) 
empties_df = pd.DataFrame(empties, columns = ['post_id'])  
df = pd.DataFrame(comments, columns = ['comment', 'score', 'post_id', 'post_title', 'author', 'sub', 'date', 'url'])

remaining_df.to_csv("remaining.csv")
empties_df.to_csv("empties.csv")
df.to_csv('reddit_dataset.csv')

#sanity check
print(len(set(df.post_id)) + len(set(remaining_df.post_id)) + len(set(empties_df.post_id)) + len(unable) == len(poststats))
