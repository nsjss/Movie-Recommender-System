import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

#print(movies.head)

movies=movies.merge(credits,on='title')
#print(movies.shape)

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
#print(movies.head)

movies.dropna(inplace=True)

def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres']=movies['genres'].apply(convert)

movies['keywords']=movies['keywords'].apply(convert)
#print(movies.head)

#print(movies['cast'][0])

def convert_cast(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        l.append(i['name'])
        if counter!=4:
            counter+=1
        else:
            break
    return l

movies['cast']=movies['cast'].apply(convert_cast)
#print(movies['cast'][0])

#print(movies['crew'][0])

def convert_cast(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l

movies['crew']=movies['crew'].apply(convert_cast)
movies['overview']=movies['overview'].apply(lambda x:x.split())
print(movies['overview'][0])

movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','')for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','')for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','')for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','')for i in x])

#print(movies.head)

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
print(new_df.head)

ps=PorterStemmer()
def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return ' '.join(l)

new_df['tags']=new_df['tags'].apply(stem)


cv=CountVectorizer(max_features=5000,stop_words='english')

vectors=cv.fit_transform(new_df['tags']).toarray()

similarity=cosine_similarity(vectors)

def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:25]

    for i in movie_list:
        print(new_df.iloc[i[0]].title,"by",str(movies.iloc[i[0]].crew))

recommend('Nightcrawler')

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
new_df['title'].values
pickle.dump(similarity,open('similarity.pkl','wb'))






