import streamlit as st
from transformers import pipeline

# Function to load the model
@st.cache_data
def load_sentiment_model():
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return sentiment_model

def main():
    st.title("Simple Sentiment Analysis App")
    st.write("This app uses a Hugging Face model to predict the sentiment of your text.")

    # Text input
    user_input = st.text_area("Enter your text here:", "I love Streamlit!")

    # Load sentiment model (cached)
    sentiment_model = load_sentiment_model()

    # Button to perform sentiment analysis
    if st.button("Analyze"):
        with st.spinner('Analyzing...'):
            result = sentiment_model(user_input)
            st.success("Done!")
            st.write("Sentiment:", result[0]['label'])
            st.write("Confidence score:", result[0]['score'])

if __name__ == "__main__":
    main()