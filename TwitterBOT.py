#!/usr/bin/env python
# coding: utf-8

# # Il Dato Mancante - BOT

# In[1]:


#magic
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# Libreria twitter
import os
import twitter
import json
import sys
import numpy as np
import pandas as pd
import logging
import logging.config
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import Counter
from urllib.error import HTTPError
from twitter import TwitterHTTPError

logging.config.fileConfig(fname='/opt/app/data/2_Twitter_Bot/logging.conf', disable_existing_loggers=False)

print('Running..')

config = {} 
config_name = r'/opt/app/data/config/config_od.py' 
exec(open(config_name).read(),config)


# Key and Secret
consumer_key=config['BOT_TWITTER_KEY']
consumer_secret=config['BOT_TWITTER_SECRET']
access_token=config['BOT_TOKEN']
access_token_secret=config['BOT_TOKEN_SECRET']


# In[3]:


# Functions
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>|\n|&nbsp|amp;|(\S+)\s*=\s*(\S+)|(\S+)\s*:\s*(\S+)|inline-block|(\S+)\s*#\s*(\S+)')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# In[4]:


def get_tweets(trend, N_STATUS):
    # Array of popular tweets of trend
    ptrends = []
    q=trend
    count = N_STATUS
    #rtype = 'popular'
    rtype = 'mixed'
    #rtype = 'recent'
    tmode = 'extended'
    lang = 'it'
    try:
        search_results = t.search.tweets(q=q, count=count, result_type=rtype, tweet_mode=tmode,
                                    lang = lang)
    except TwitterHTTPError as e:
        print('Error: ', e)
        sys.exit(0)
    for result in search_results['statuses']:
        dtrends = {}
        dtrends.update(trend=trend)
        dtrends.update(id=result['id'])
        dtrends.update(created_at=result['created_at'])
        dtrends.update(text=result['full_text'])
        dtrends.update(user_id=result['user']['id'])
        dtrends.update(user_name=result['user']['screen_name'])
        dtrends.update(user_followers=result['user']['followers_count'])
        dtrends.update(like=result['favorite_count'])
        dtrends.update(retweet=result['retweet_count'])
        ptrends.append(dtrends)
    return ptrends


# In[5]:


def get_discussions(user, tweet_id, N_REPLY):
# Array of discussion of popular tweet by id
    pdiscussion = []
    max_id = None
    tmode = 'extended'
    lang = 'it'
    q='to:'+user
    try:
        replies = t.search.tweets(q=q,since_id=tweet_id, max_id=max_id, count=N_REPLY,tweet_mode=tmode,
                             lang=lang)
    except TwitterHTTPError as e:
        print('Error: ', e)
        sys.exit(0)
    for reply in replies['statuses']:
        dreply = {}
        if reply['in_reply_to_status_id'] == tweet_id:
            dreply.update(id_popular=tweet_id)
            dreply.update(id=reply['id'])
            dreply.update(created_at=reply['created_at'])
            dreply.update(text=reply['full_text'])
            dreply.update(user_id=reply['user']['id'])
            dreply.update(user_name=reply['user']['screen_name'])
            dreply.update(user_followers=reply['user']['followers_count'])
            dreply.update(like=reply['favorite_count'])
            dreply.update(retweet=reply['retweet_count'])
            pdiscussion.append(dreply)
    return pdiscussion


# In[6]:


# Authentication
auth = twitter.oauth.OAuth(access_token, access_token_secret,
                           consumer_key, consumer_secret)
t = twitter.Twitter(auth=auth)


# In[7]:


# http://woeid.rosselliot.co.nz/lookup/roma
ITALIA = config['BOT_PLACE']
N_TRENDS = config['N_TRENDS']
N_STATUS = config['N_STATUS']
N_REPLY = config['N_REPLY']


# In[8]:


# Read the data of posts
filename = r'/opt/app/data/2_Twitter_Bot/wp_posts.pkl'
df_posts = pd.read_pickle(filename)


# In[9]:


# Leggo i trend in Italia
try:
    ita_trends = t.trends.place(_id=ITALIA)
except TwitterHTTPError as e:
        print('Error: ', e)
        sys.exit(0)
trends = json.loads(json.dumps(ita_trends, indent=1))
trends_arr = []
for trend in trends[0]["trends"]:
    trends_dic = {}
    trends_dic['trend'] = trend['name']
    trends_arr.append(trends_dic)


# In[10]:


# Avvio BOT su Hastag ad hoc, metti rtype=recent su get_tweets
#trends_arr = []
#trends_arr.append({'trend': '#dop'})
#trends_arr.append({'trend': '#sentiment'})
#trends_arr.append({'trend': '#juarez'})
#trends_arr.append({'trend': '#slowfood'})
#trends_arr.append({'trend': '#co2'})
#trends_arr.append({'trend': '#opendata'})
#trends_arr.append({'trend': '#rifugiati'})
#trends_arr.append({'trend': '#igp'})
#trends_arr.append({'trend': '#pat'})


# In[11]:


# Per ogni trend recupero i tweet popolari
tws = []
for trend in trends_arr[0:N_TRENDS]:
    ptrend = trend['trend']
    logging.info('TREND: {}'.format(ptrend))
    tpopular = get_tweets(ptrend, N_STATUS)
    # Per ogni tweet popolare recupero le discussioni
    for pop in tpopular:
        user = pop['user_name']
        tweet_id = pop['id']
        pop_text = pop['text'].replace('|', ' ')
        pop_text = pop_text.replace('\n', ' ')
        tdiscussion = get_discussions(str(user),tweet_id, N_REPLY)
        tw_text = pop['text'] + '\n' + pop['trend'] +                     '\n'.join([d['text'] for d in tdiscussion])
        documents = np.append(df_posts[['content']].values,tw_text).ravel()  # Creo la lista dei documenti
        n_clusters = len(documents)-1 # Numero di cluster = articoli, fuori la discussione twitter
        # Creo la pipeline
        pipeline = Pipeline([('feature_extraction', TfidfVectorizer(max_df=0.4)),
                     ('clusterer', KMeans(n_clusters=n_clusters))
                     ])
        pipeline.fit(documents)
        labels = pipeline.predict(documents)
        # Estraggo i termini
        terms = pipeline.named_steps['feature_extraction'].get_feature_names()
        cluster_number = labels[len(documents)-1]
        centroid = pipeline.named_steps['clusterer'].cluster_centers_[cluster_number]
        most_important = centroid.argsort()
        # Check if twitter discussion is in a cluster with an article
        # Twitter discussion is the last
        if(len(np.where(labels==labels[len(documents)-1])[0]))>1:
            # Se è in cluster prendi il primo, che ultimo è sicuro articolo
            post_index = np.where(labels==labels[len(documents)-1])[0][0]
            url = df_posts[['link']].values[post_index][0]
            # invia tweet in risposta a popular
            status = 'Ciao '+'@'+tpopular[0]['user_name']+' sono un bot, potrei sbagliarmi, forse ti interessa... ' + str(ptrend) + ' ' + url
            try:
                tweet = t.statuses.update(status=status,in_reply_to_status_id=tpopular[0]['id'])
                logging.info('Cluster SI articolo: {} post: {} topic: {}, {}, {}, {}, {} tweet: {}, tweet_pop: {} '.format(df_posts[['title']].values[post_index][0],pop_text,terms[most_important[-1]],terms[most_important[-2]], terms[most_important[-3]],terms[most_important[-4]], terms[most_important[-5]], tweet['id'], tpopular[0]['id']))
            except TwitterHTTPError as e:
                logging.error('Errore, tweet non inviato: {}'.format(e)) 
        else:
            logging.info('Cluster NO post: {} topic: {}, {}, {}, {}, {}'.format(pop_text,terms[most_important[-1]], terms[most_important[-2]], terms[most_important[-3]], terms[most_important[-4]], terms[most_important[-5]]))

