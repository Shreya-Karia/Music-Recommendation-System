#!/usr/bin/env python
# coding: utf-8

# In[7]:


#I have used the spotify API for developers
get_ipython().system('pip install spotipy')


# In[8]:


#getting all the libraries required
import pandas as pd
import numpy as np
import json
import re
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import warnings
warnings.filterwarnings("ignore")


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


#reading the datasets the dataset


# In[11]:


spotify_df = pd.read_csv("C:/Users/shrey/Desktop/Engage/data/data.csv")


# In[12]:


spotify_df.head()


# In[13]:


data_w_genre = pd.read_csv("C:/Users/shrey/Desktop/Engage/data/data_w_genres.csv")
data_w_genre.head()


# In[14]:


data_w_genre.dtypes


# In[15]:


#converting datatype of coulmn genres according to our requirement


# In[16]:


data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])


# In[17]:


data_w_genre['genres_upd'].values[0][0]


# In[18]:


spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))


# In[19]:


spotify_df['artists'].values[0]


# In[20]:


spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)


# In[21]:


spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )


# In[22]:


spotify_df['artists_song'] = spotify_df.apply(lambda row: str(row['artists_upd'][0])+str(row['name']),axis = 1)


# In[23]:


spotify_df[spotify_df['name']=='Adore You']


# In[24]:


#dropping off duplicates


# In[25]:


spotify_df.drop_duplicates('artists_song',inplace = True)


# In[26]:


spotify_df[spotify_df['name']=='Adore You']


# In[27]:


artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')


# In[28]:


#getting genres in usauble forms and then merging ṭhem


# In[29]:


artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')


# In[30]:


artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]


# In[31]:


artists_exploded_enriched_nonnull


# In[32]:


artists_exploded_enriched_nonnull[artists_exploded_enriched_nonnull['id'] =='3jjujdWJ72nww5eGnfs2E7']


# In[33]:


artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()


# In[34]:


#getting genres in a list along with corresponding songs
artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))
artists_genres_consolidated.head()


# In[35]:


#merging to get a single dataset
spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')


# In[36]:


spotify_df


# In[37]:


#converting values in ohe variables: popularity and year 


# In[38]:


spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])


# In[39]:


float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values


# In[40]:


ohe_cols = 'popularity'


# In[41]:


spotify_df['popularity'].describe()


# In[42]:


spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))


# In[43]:


spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda m: m if isinstance(m,list) else[])


# In[44]:


spotify_df.head()


# In[45]:


def ohe_cre(df, column, new):
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)
    return tf_df


# In[46]:


#Using TFIDF to generate vectors to use for recommendations
def feature_set(df, float_cols):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)
    
    year_ohe = ohe_cre(df, 'year', 'year') * 0.5
    popularity_ohe = ohe_cre(df,'popularity_red','pop') * 0.15
    
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2
    
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
    
    final['id'] = df['id'].values
    
    return final


# In[47]:


#This feature set has values in form of numbers which will be used to genrate recommendations
complete_feature_set = feature_set(spotify_df, float_cols = float_cols)


# In[48]:


complete_feature_set.head()


# In[49]:


#Connecting to spotify api to get playlists


# In[50]:


client_id = '403fa2a972534d239b5cb78b46a2662d'
client_secret = '6bb32251ab114b04b74fdee713fe1944'


# In[51]:


scope = 'user-library-read'

if len(sys.argv) > 1:
    username = sys.argv[1]
else:
    print("Usage: %s username" % (sys.argv[0],))
    sys.exit()


# In[52]:


auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret = client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)


# In[53]:


token = util.prompt_for_user_token(scope, client_id = client_id, client_secret=client_secret, redirect_uri='http://localhost:8881/') 


# In[54]:


sp = spotipy.Spotify(auth=token)


# In[55]:


#getting playlists to use them as input to get similiar playlists
id_name = {}
list_photo = {}
for i in sp.current_user_playlists()['items']:

    id_name[i['name']] = i['uri'].split(':')[2]
    #list_photo[i['uri'].split(':')[2]] = i['images'][0]['url']


# In[56]:


id_name


# In[57]:


#a function which will take in names of playlists and use them to get name of all the songs in there
def get_songs(playlist_name,id_dic,df):
    playlist = pd.DataFrame()
    playlist_name = playlist_name
    
    for ix, i in enumerate(sp.playlist(id_dic[playlist_name])['tracks']['items']):
        #print(i['track']['artists'][0]['name'])
        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    
    return playlist


# In[58]:


#getting songs of one playlist
playlist_name = get_songs('YES!',id_name,spotify_df)


# In[59]:


playlist_name


# In[60]:


from skimage import io
import matplotlib.pyplot as plt
#takes in the songs of a dataframes and diaplays songs' cover images
def visualize_songs(df):
    
    temp = df['url'].values
    plt.figure(figsize=(15,int(0.625* len(temp))))
    columns = 5
    
    for i, url in enumerate(temp):
        plt.subplot(len(temp) / columns + 1, columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks(color = 'w', fontsize = 0.1)
        plt.yticks(color = 'w', fontsize = 0.1)
        plt.xlabel(df['name'].values[i], fontsize = 12)
        plt.tight_layout(h_pad=0.4, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()


# In[61]:


visualize_songs(playlist_name)


# In[62]:


#creates a playlist vector from its feature set
def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):
    
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added', ascending = False)
    
    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:,:-4]
    
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist


# In[63]:


complete_feature_set_playlist_vector_name, complete_feature_set_nonplaylist_name = generate_playlist_feature(complete_feature_set, playlist_name, 1.09)


# In[64]:


complete_feature_set_playlist_vector_name.shape


# In[65]:


#using sine cosine similarity it generates a playlist of recommended songs
def generate_playlist_recomm(df, features, nonplaylist_features):
    
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_10 = non_playlist_df.sort_values('sim',ascending = False).head(10)
    non_playlist_df_top_10['url'] = non_playlist_df_top_10['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    
    return non_playlist_df_top_10


# In[69]:


Final_top10 = generate_playlist_recomm(spotify_df, complete_feature_set_playlist_vector_name, complete_feature_set_nonplaylist_name)


# In[67]:


Final_top10


# In[68]:


#generated playlist one would like
visualize_songs(Final_top10)

