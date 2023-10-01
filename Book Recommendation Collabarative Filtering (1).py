import pandas as pd
import numpy as np
books = pd.read_csv("C:/Users/BALU/Downloads/Deployment Book recommendation system/Books.csv")
users = pd.read_csv("C:/Users/BALU/Downloads/Deployment Book recommendation system/Users.csv")
rating = pd.read_csv("C:/Users/BALU/Downloads/Deployment Book recommendation system/Ratings.csv")
books.isnull().sum()
rating.isnull().sum()
users.isnull().sum()
rating_with_name = rating.merge(books,on='ISBN')
rating_with_name.drop(columns=["Image-URL-S","Image-URL-M","Image-URL-L"],axis=1, inplace=True)
rating_with_name.isnull().sum()
rating_with_name.loc[rating_with_name['Book-Author'].isnull(),:]
rating_with_name.at[863398,'Book-Author'] = 'Other'
rating_with_name.loc[rating_with_name['Publisher'].isnull(),:]
rating_with_name.at[862973,'Publisher'] = 'Other'
rating_with_name.at[862984,'Publisher'] = 'Other'
rating_with_name.isnull().sum()
rating_with_name1 = rating_with_name.reset_index(drop=True)
rating_with_name1.drop(columns=["ISBN"], index=1, inplace=True)
rating_with_name1.sort_values("User-ID")
num_rating_df = rating_with_name1.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)
avg_rating_df = rating_with_name1.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_ratings'}, inplace=True)
popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
top_50 = popularity_df[popularity_df['num_ratings']>=250].sort_values("avg_ratings",ascending=False).head(50)
final_top_50 = top_50.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Publisher','num_ratings','avg_ratings']]
x = rating_with_name1.groupby('User-ID').count()['Book-Rating']>200
knowledgable_users = x[x].index
filtered_rating = rating_with_name1[rating_with_name1['User-ID'].isin(knowledgable_users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index
final_ratings =  filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID'
                          ,values='Book-Rating')
pt
pt.fillna(0,inplace=True)
pt
from sklearn.metrics.pairwise import cosine_similarity
similarity_score = cosine_similarity(pt)
similarity_score.shape
def recommend(book_name):
    index = np.where(pt.index==book_name)[0][0]
    similar_books = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1], reverse=True)[1:6]
    
    data = []
    
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
           
        data.append(item)
    return data
recommend("Message in a Bottle")
import streamlit as st
# Streamlit UI
def main():
    st.title("Book Recommendation Engine")
    book_name = st.text_input("Enter a book name:")
    if st.button("Recommend"):
        if book_name:
            recommendations = recommend(book_name)
            for rec in recommendations:
                st.write("Title:", rec[0])
                st.write("Author:", rec[1])
                st.write("Link:", rec[2])

if __name__ == "__main__":
    main()
            

    

