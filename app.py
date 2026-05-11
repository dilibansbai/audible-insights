import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import recommend_content,recommend_cluster,recommend_hybrid
from sklearn.cluster import KMeans

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_books_data.csv")

df = load_data()

@st.cache_resource
def create_model(df):
    df = df.reset_index(drop=True)
    
    df['title'] = df['title'].fillna('')
    df['author'] = df['author'].fillna('')
    df['description'] = df['description'].fillna('')
    
    df['content'] = df['title'] + " " + df['author'] + " " + df['description']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['content'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return cosine_sim, df
    

cosine_sim, df = create_model(df)

print("DF length:", len(df))
print("Cosine shape:", cosine_sim.shape)

st.title("Audible Insights - Book Recommendation System")

st.write("Get personalized book recommendations using AI/ML")

book_list = df['title'].unique()

selected_book = st.selectbox("Choose a book", book_list)

def recommend_books_ui(book_title):
    book_title = book_title.lower()
    
    matches = df[df['title'].str.contains(book_title)]
    
    if matches.empty:
        return []
    
    idx = matches.index[0]
    
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    
    indices = [i[0] for i in scores]
    
    return df.iloc[indices][['title', 'author', 'rating']]

# if st.button("Recommend"):
#     results = recommend_books_ui(selected_book)
    
#     if len(results) == 0:
#         st.write("No recommendations found")
#     else:
#         st.subheader("Recommended Books:")
#         for i, row in results.iterrows():
#             st.write(f"{row['title']} | {row['author']} | {row['rating']}")

model_type = st.radio(
    "Choose Recommendation Type:",
    ["Content-Based", "Cluster-Based", "Hybrid"]
)

if st.button("Recommend"):
    if model_type == "Content-Based":
        results = recommend_content(df,cosine_sim,selected_book)
    elif model_type == "Cluster-Based":
        results = recommend_cluster(df,selected_book)
    else:
        results = recommend_hybrid(df,cosine_sim,selected_book)
        
    if results.empty:
        st.warning("No recommendations found")
    else:
        st.dataframe(results)        