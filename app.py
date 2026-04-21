import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Bookends UAE",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CYAN THEME ====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e6f9fb 0%, #ffffff 100%);
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem;
    background: #00bcd4;
    border-radius: 20px;
    margin-bottom: 2rem;
}

.main-header h1 {
    color: white;
    margin: 0;
    font-size: 2.5rem;
}

.main-header p {
    color: #e0f7fa;
}

/* Cards */
.story-card {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    margin: 0.8rem 0;
    border: 1px solid #b2ebf2;
}

/* Buttons */
.stButton > button {
    background: #00bcd4;
    color: white;
    border-radius: 25px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #00acc1;
}

[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== SAFE LOGO ====================
if os.path.exists("bookends_logo.png"):
    st.image("bookends_logo.png", width=150)
else:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Logo_placeholder.png", width=150)

# ==================== HEADER ====================
st.markdown("""
<div class="main-header">
<h1>Bookends UAE</h1>
<p>Your Smart Book Recommender</p>
</div>
""", unsafe_allow_html=True)

# ==================== DATA ====================
def create_books():
    titles = [f"Book {i}" for i in range(1, 81)]
    authors = [f"Author {i%10}" for i in range(1, 81)]
    genres = ["fiction","fantasy","romance","sci-fi","business","self-help","classic","thriller"] * 10
    
    df = pd.DataFrame({
        "Book Title": titles,
        "Author": authors,
        "Genre": genres[:80]
    })
    
    df["combined"] = df["Book Title"] + " " + df["Author"] + " " + df["Genre"]
    return df

books = create_books()

# ==================== RECOMMENDER ====================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined'])
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend_by_title(title):
    idx = books[books['Book Title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return books.iloc[[i[0] for i in scores]]

def recommend_by_genre(genre):
    return books[books['Genre'] == genre].sample(5)

def vibe_recommend(text):
    return books.sample(5)

# ==================== FAQ ====================
FAQ = {
    "location": "Dubai Digital Park, Silicon Oasis",
    "delivery": "Free delivery above AED 180",
    "hours": "10 AM - 10 PM daily",
    "sell books": "Yes, you can sell your books at Bookends UAE."
}

def faq_answer(q):
    q = q.lower()
    for key in FAQ:
        if key in q:
            return FAQ[key]
    return "Ask about location, delivery, or hours."

# ==================== DISPLAY ====================
def display_books(df):
    for _, row in df.iterrows():
        st.markdown(f"""
        <div class="story-card">
        📚 <b>{row['Book Title']}</b><br>
        ✍️ {row['Author']}<br>
        🏷️ {row['Genre']}
        </div>
        """, unsafe_allow_html=True)

# ==================== SIDEBAR ====================
menu = st.sidebar.radio("Menu", [
    "Home",
    "Find Books",
    "Chatbot",
    "Dashboard"
])

# ==================== HOME ====================
if menu == "Home":
    st.write("### Welcome to Bookends UAE 📚")
    st.write(f"Total Books: {len(books)}")

# ==================== FIND BOOKS ====================
elif menu == "Find Books":
    tab1, tab2, tab3 = st.tabs(["By Genre", "By Title", "By Mood"])
    
    with tab1:
        genre = st.selectbox("Select Genre", books['Genre'].unique())
        if st.button("Recommend Genre"):
            display_books(recommend_by_genre(genre))
    
    with tab2:
        title = st.selectbox("Select Book", books['Book Title'])
        if st.button("Recommend Similar"):
            display_books(recommend_by_title(title))
    
    with tab3:
        mood = st.text_input("Describe mood")
        if st.button("Find by Mood"):
            display_books(vibe_recommend(mood))

# ==================== CHATBOT ====================
elif menu == "Chatbot":
    st.write("### Ask a Question")
    q = st.text_input("Type here")
    if st.button("Ask"):
        st.write(faq_answer(q))

# ==================== DASHBOARD ====================
elif menu == "Dashboard":
    st.write("### Genre Distribution")
    st.bar_chart(books['Genre'].value_counts())
