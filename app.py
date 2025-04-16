import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import requests
import difflib

# ✅ Load data
@st.cache_data
def load_data():
    df_movies = pd.read_csv(r"C:\Users\MF'\Downloads\AI_project\clean_movie_dataset.csv", encoding="utf-8")
    return df_movies

df_movies = load_data()
#df_movies['combined_features'] =  df_movies[['genres', 'cast', 'director', 'overview']].agg(lambda x: ' '.join(x.replace('unknown', '') for x in x), axis=1)
@st.cache_data
def extract_features(df):
    """Convert text & numeric features into a combined feature matrix."""
    
    # ✅ 1️⃣ Convert text features using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    text_features = tfidf.fit_transform(df['combined_features'].fillna(''))

    # ✅ 2️⃣ Normalize numeric features using MinMaxScaler
    scaler = MinMaxScaler()
    numeric_features = df[['vote_average', 'budget', 'popularity']].fillna(0)
    numeric_scaled = scaler.fit_transform(numeric_features)

    # ✅ 3️⃣ Combine all features into a single matrix
    feature_matrix = np.hstack((text_features.toarray(), numeric_scaled))

    return feature_matrix

def train_cosine_model(df):
    feature_matrix = extract_features(df)  # ✅ Reuse function
    similarity_matrix = cosine_similarity(feature_matrix)
    return similarity_matrix
similarity_matrix = train_cosine_model(df_movies)

# ✅ TMDb API setup
API_KEY = "3967b3a465c7a7f013550ae800ea4c48"
BASE_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
FREE_STREAMING_URL = "https://www.justwatch.com/us/search?q={}"  # Free streaming search
CERTIFICATION_URL = "https://api.themoviedb.org/3/movie/{}/release_dates"

# ✅ Get closest movie match
def find_closest_movie(movie_title):
    choices = df_movies['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_title, choices)
    return find_close_match[0] if find_close_match else None

# ✅ Fetch movie details (poster & streaming link)
def get_movie_details(movie_title):
    params = {"api_key": API_KEY, "query": movie_title}
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if data["results"]:
        for movie in data["results"]:
            movie_id = movie["id"]
            # ✅ Check if the movie is appropriate (G, PG, PG-13)
            cert_response = requests.get(CERTIFICATION_URL.format(movie_id), params={"api_key": API_KEY})
            cert_data = cert_response.json()

            if "results" in cert_data:
                for country in cert_data["results"]:
                    if country["iso_3166_1"] == "US":  # Check US rating
                        for release in country["release_dates"]:
                            rating = release["certification"]
                            if rating in ["G", "PG", "PG-13"]:  # Acceptable ratings
                                poster_url = IMAGE_BASE_URL + movie["poster_path"] if movie.get("poster_path") else "https://via.placeholder.com/500x750?text=No+Image"
                                watch_link = FREE_STREAMING_URL.format(movie_title)
                                return poster_url, watch_link
    return None, None  # Return None if no appropriate movie found

# ✅ Get Recommendations
def get_recommendations(movie_title, top_n=10):
    matched_movie = find_closest_movie(movie_title)
    if not matched_movie:
        return None  # No match found

    idx = df_movies[df_movies['title'] == matched_movie].index[0]
    sim_scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)[0:top_n]
    recommended_movie_indices = [i[0] for i in sim_scores]
    
    recommendations = df_movies.iloc[recommended_movie_indices][['title']]
    recommendations[['poster_path', 'watch_link']] = recommendations['title'].apply(lambda x: pd.Series(get_movie_details(x)))
    return recommendations.dropna()  # Drop movies without valid images or links

# ✅ Streamlit Interface
st.title("🎬 Movie Recommendation System")

# 🔝 Show Top Popular Movies
st.write("## 🔝 Most Popular Movies")

if 'popularity' in df_movies.columns:
    top_popular_movies = df_movies.sort_values(by='popularity', ascending=False)[['title']].head(10)
    popular_movie = st.selectbox("Choose a popular movie to get recommendations:", top_popular_movies['title'])
else:
    st.error("⚠️ Popularity data not available.")
    popular_movie = None

# 🔎 Search box
movie_name = st.text_input("Or enter a movie name:", "")

if st.button("Get Recommendations"):
    selected_movie = movie_name if movie_name else popular_movie
    if selected_movie:
        recommendations = get_recommendations(selected_movie, 10)
        if recommendations is not None and not recommendations.empty:
            st.success(f"✅ Showing results for: {find_closest_movie(selected_movie)}")
            st.write("### Recommended Movies:")

            # ✅ Display recommendations in rows of 5
            rows = [recommendations.iloc[i:i+5] for i in range(0, len(recommendations), 5)]
            for row in rows:
                cols = st.columns(len(row))
                for col, (_, movie) in zip(cols, row.iterrows()):
                    with col:
                        st.image(movie['poster_path'], width=150)
                        st.markdown(f"[🎥 {movie['title']}]({movie['watch_link']})", unsafe_allow_html=True)
        else:
            st.error("⚠️ No appropriate recommendations found.")
    else:
        st.error("⚠️ Please select or enter a movie name.")
