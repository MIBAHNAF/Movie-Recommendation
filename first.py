import pandas as pd
import re
import stdio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_path_1 ='E:\Movie Recommnedation\ml-25m\movies.csv'
movies = pd.read_csv(data_path_1)

data_path_2 = "E:\Movie Recommnedation\ml-25m\datings.csv"
ratings = pd.read_csv(data_path_2)

def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]","",title)

movies["clean_title"] = movies["title"].apply(clean_title)



vectorizer = TfidfVectorizer(ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])



def search(title):
    title = clean_title(title)
    search_vec = vectorizer.transform([title])
    similarity = cosine_similarity(search_vec,tfidf).flatten()
    indices = np.argpartition(similarity,-5)[-5:]
    results = movies.iloc[indices][::-1]
    return results


def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"]==movie_id) & (ratings["rating"]>=4 )]["userId"].unique()
    similar_users_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>=4 )]["movieId"]

    similar_users_recs = similar_users_recs.value_counts() / len(similar_users)
    similar_users_recs = similar_users_recs[similar_users_recs> .20]

    all_users =ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["rating"] >=4)]
    all_users_recs = all_users["movieId"].value_counts() /len(all_users["userId"].unique())

    rec_percent =pd.concat([similar_users_recs,all_users_recs],axis =1)
    rec_percent.columns = ["similar","all"]

    rec_percent["score"] = rec_percent["similar"] / rec_percent["all"]

    rec_percent  = rec_percent.sort_values("score", ascending=False)
    return rec_percent.head(5).merge(movies,left_index= True, right_on ="movieId")[["score","title","genres"]]

stdio.write("Movie Title : ")
results =search(input())
movie_id = results.iloc[0]["movieId"]
display = find_similar_movies(movie_id)
print(display)






