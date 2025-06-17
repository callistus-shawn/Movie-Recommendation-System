#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

class mov:
 def rec(self,title):
  cred=pd.read_csv("tmdb_5000_credits.csv")
  mov=pd.read_csv("tmdb_5000_movies.csv")
  mov.head()
  cred.head()
  cred=cred.rename(columns={"movie_id":"id"})
  merge=mov.merge(cred,on='id')
  tfv=TfidfVectorizer(strip_accents="unicode",analyzer="word",token_pattern="\w{1,}",stop_words="english")
  merge["overview"].fillna(" ",inplace=True)
  mat=tfv.fit_transform(merge["overview"])

  sig=sigmoid_kernel(mat,mat)
  indices=pd.Series(merge.index,index=merge["original_title"]
                 ).drop_duplicates()

 
  ind=indices[title]
  score=sorted(list(enumerate(sig[ind])),key=lambda x:x[1],reverse=True)
  moviescore=[i[0] for i in score[1:9]]
  return list(merge["original_title"].iloc[moviescore])

r=mov()
with open('model.pkl', 'wb') as f:
    pickle.dump(r, f)



