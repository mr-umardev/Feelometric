import streamlit as st
import sqlite3
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Download stopwords if not already done
nltk.download('stopwords')
stop_words = stopwords.words('english')

DATABASE = 'text_analysis.db'

# Initialize SQLite database with updated column names
def init_db():
    """Initialize the SQLite database and create the table with proper column names."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS TextEntries (
            original_text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            sentiment_score REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Store text and sentiment in the database with updated column names
def store_in_db(original_text, processed_text, sentiment_score):
    """Store the user-entered text, processed text, and sentiment score in the SQLite database."""
    if original_text:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO TextEntries (original_text, processed_text, sentiment_score)
            VALUES (?, ?, ?)
        ''', (original_text, processed_text, sentiment_score))
        conn.commit()
        conn.close()

# Preprocess the input text by removing digits and stopwords
def preprocess_text(text):
    text_final = ''.join(c for c in text if not c.isdigit())  # Remove digits
    processed_text = ' '.join([word for word in text_final.split() if word not in stop_words])
    return processed_text

# Function to get video file in base64
def get_video_base64(video_path):
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
        return base64.b64encode(video_bytes).decode()

# Main function for Streamlit App
def main():
    # Set background video (always render the video in the background)
    video_path = Path("static/videos/PurpleStar.mp4")  # Path to the video
    video_b64 = get_video_base64(video_path)
    video_html = f"""
    <video autoplay loop muted style="position: fixed; width: 100vw; height: 100vh; object-fit: cover; top: 0; left: 0; z-index: -1;">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    
    # Add the background video
    st.markdown(video_html, unsafe_allow_html=True)

    # Overlay (for adding text on top of video)
    st.markdown("""
        <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); z-index: 0;"></div>
    """, unsafe_allow_html=True)

    # Set the Streamlit UI without the default black background
    st.markdown("""
        <style>
            body {
                background-color: transparent;
            }
            .stApp {
                background: transparent;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Feelometric.")

    # Input text
    text = st.text_area("Enter text to analyze:", "")

    if st.button("Submit"):
        if text:
            # Preprocess text
            processed_text = preprocess_text(text)

            # Sentiment analysis using VADER
            sa = SentimentIntensityAnalyzer()
            sentiment = sa.polarity_scores(processed_text)
            compound = round((1 + sentiment['compound']) / 2, 2)

            # Prepare the results in a table format (excluding Sentiment Score)
            sentiment_data = {
                "Sentiment Type": ["Positive", "Negative", "Neutral"],
                "Score": [sentiment['pos'], sentiment['neg'], sentiment['neu']]
            }

            # Create a DataFrame from the sentiment data
            sentiment_df = pd.DataFrame(sentiment_data)

            # Display the sentiment results as a table
            st.markdown(f'<h3 style="color: yellow;">Sentiment Results (Sentiment Score: {compound})</h3>', unsafe_allow_html=True)
            st.dataframe(sentiment_df)

            # Store results in the database with updated column names
            store_in_db(text, processed_text, compound)

            # Show confirmation message
            st.success("Sentiment score and text stored in the database!")

    # Button for "Show Stored Database"
    if st.button("Show Stored Database"):
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT original_text, processed_text, sentiment_score FROM TextEntries')
        rows = cursor.fetchall()
        conn.close()

        if rows:
            # Display the rows in a table with the correct column names
            df = pd.DataFrame(rows, columns=["Original Text", "Processed Text", "Sentiment Score"])
            st.write("Stored Entries:")
            st.dataframe(df)

        else:
            st.write("No data available in the database")

    # Button for "Visualize Sentiments"
    if st.button("Visualize Sentiments"):
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT sentiment_score FROM TextEntries')
        rows = cursor.fetchall()
        conn.close()

        if rows:
            # Create a DataFrame from the sentiment scores
            sentiment_scores = [row[0] for row in rows]
            df = pd.DataFrame(sentiment_scores, columns=["Sentiment Score"])

            # Plot the line chart for Sentiment Scores over the entries
            st.markdown("### Sentiment Score Over Time")
            st.line_chart(df)

        else:
            st.write("No data available in the database")

    # About section
    if st.button("About"):
        st.write("This app analyzes text sentiment using VADER and stores results in an SQLite database.")

if __name__ == "__main__":
    init_db()  # Initialize database
    main()