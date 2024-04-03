import streamlit as st
from transformers import pipeline

# Caching the model loading to improve performance and reduce memory usage
@st.cache_data
def load_model():
    # Load the summarization model
    model = pipeline("summarization", model="facebook/bart-large-cnn")
    return model

def main():
    st.title("AI-Powered Text Summarizer")
    st.subheader("Summarize long articles into concise summaries with AI")
    
    # Instructions for the user
    st.write("Please enter the text you want to summarize (up to 500 words for optimal performance):")
    
    # User input for the text to be summarized
    user_input_text = st.text_area("Text to summarize:", height=300)
    
    # Button to trigger summarization
    if st.button("Summarize"):
        if user_input_text:
            # Limiting input size for memory considerations (optional, based on your testing)
            if len(user_input_text.split()) > 500:
                st.warning("The text is too long and might exceed resource limits. Please try with shorter text.")
                return

            try:
                # Generating the summary with error handling
                summarizer = load_model()
                summary = summarizer(user_input_text, max_length=130, min_length=30, do_sample=False)
                st.write("Summary:")
                st.write(summary[0]['summary_text'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try again with shorter text or different content.")
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
