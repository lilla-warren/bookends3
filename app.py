import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Bookends UAE",
    page_icon="📚",
    layout="wide"
)

# ==================== NEW THEME ====================
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
    font-size: 2.5rem;
    margin: 0;
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

# ==================== DATA ====================
def create_books():
    data = {
        'Book Title': [
            'Atomic Habits','Deep Work','The Alchemist','1984','Dune',
            'Harry Potter 1','Harry Potter 2','Harry Potter 3',
            'The Hobbit','Lord of the Rings','The Great Gatsby',
            'To Kill a Mockingbird','The Catcher in the Rye',
            'Pride and Prejudice','Jane Eyre','Wuthering Heights',
            'The Book Thief','The Kite Runner','A Thousand Splendid Suns',
            'Sapiens','Homo Deus','Educated','Becoming',
            'Rich Dad Poor Dad','Think and Grow Rich','Psychology of Money',
            'Zero to One','Start with Why','Lean Startup',
            'Subtle Art of Not Giving a F*ck',
            'It Ends With Us','It Starts With Us',
            'Verity','Ugly Love','November 9',
            'Twilight','New Moon','Eclipse','Breaking Dawn',
            'Hunger Games','Catching Fire','Mockingjay',
            'Percy Jackson 1','Percy Jackson 2','Percy Jackson 3',
            'Little Prince','Alice in Wonderland',
            'Narnia','Fault in Our Stars',
            'Looking for Alaska','Paper Towns',
            'Maze Runner','Scorch Trials','Death Cure',
            'Da Vinci Code','Angels and Demons',
            'Inferno','Origin'
        ],
        'Author': [
            'James Clear','Cal Newport','Paulo Coelho','George Orwell','Frank Herbert',
            'J.K. Rowling','J.K. Rowling','J.K. Rowling',
            'Tolkien','Tolkien','Fitzgerald',
            'Harper Lee','Salinger',
            'Jane Austen','Charlotte Bronte','Emily Bronte',
            'Zusak','Hosseini','Hosseini',
            'Harari','Harari','Westover','Obama',
            'Kiyosaki','Napoleon Hill','Housel',
            'Thiel','Sinek','Ries',
            'Mark Manson',
            'Colleen Hoover','Colleen Hoover',
            'Colleen Hoover','Colleen Hoover','Colleen Hoover',
            'Stephenie Meyer','Stephenie Meyer','Stephenie Meyer','Stephenie Meyer',
            'Suzanne Collins','Suzanne Collins','Suzanne Collins',
            'Rick Riordan','Rick Riordan','Rick Riordan',
            'Saint-Exupery','Lewis Carroll',
            'C.S. Lewis','John Green',
            'John Green','John Green',
            'Dashner','Dashner','Dashner',
            'Dan Brown','Dan Brown',
            'Dan Brown','Dan Brown'
        ],
        'Genre': [
            'self-help','productivity','fiction','fiction','sci-fi',
            'fantasy','fantasy','fantasy',
            'fantasy','fantasy','classic',
            'classic','classic',
            'romance','classic','classic',
            'fiction','fiction','fiction',
            'history','history','memoir','memoir',
            'finance','self-help','finance',
            'business','business','business',
            'self-help',
            'romance','romance',
            'romance','romance','romance',
            'fantasy','fantasy','fantasy','fantasy',
            'sci-fi','sci-fi','sci-fi',
            'fantasy','fantasy','fantasy',
            'children','fantasy',
            'fantasy','romance',
            'romance','romance',
            'sci-fi','sci-fi','sci-fi',
            'thriller','thriller',
            'thriller','thriller'
        ]
    }
    df = pd.DataFrame(data)
    df['combined'] = df['Book Title'] + " " + df['Author'] + " " + df['Genre']
    return df

books = create_books()

# ==================== RECOMMENDER ====================
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(books['combined'])
cosine_sim = cosine_similarity(matrix)

def recommend_by_title(title):
    idx = books[books['Book Title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [books.iloc[i[0]]['Book Title'] for i in scores]

# ==================== FAQ ====================
FAQ = {
    "location": "Dubai Digital Park, Silicon Oasis",
    "delivery": "Free above AED 180",
    "hours": "10am - 10pm daily"
}

def faq_answer(q):
    q = q.lower()
    for key in FAQ:
        if key in q:
            return FAQ[key]
    return "Try asking about location, delivery or hours."

# ==================== HEADER ====================
st.image("bookends_logo.png", width=150)

st.markdown("""
<div class="main-header">
<h1>Bookends UAE</h1>
<p>Your smart book recommender</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
menu = st.sidebar.radio("Menu", [
    "Home",
    "Recommend",
    "Chatbot",
    "Dashboard"
])

# ==================== HOME ====================
if menu == "Home":
    st.write("### Welcome to Bookends UAE 📚")
    st.write(f"Total Books: {len(books)}")

# ==================== RECOMMENDER ====================
elif menu == "Recommend":
    title = st.selectbox("Choose a book", books['Book Title'])
    
    if st.button("Recommend"):
        recs = recommend_by_title(title)
        for r in recs:
            st.markdown(f"<div class='story-card'>📚 {r}</div>", unsafe_allow_html=True)

# ==================== CHATBOT ====================
elif menu == "Chatbot":
    q = st.text_input("Ask a question")
    if st.button("Ask"):
        st.write(faq_answer(q))

# ==================== DASHBOARD ====================
elif menu == "Dashboard":
    st.write("### Genre Distribution")
    st.bar_chart(books['Genre'].value_counts())
