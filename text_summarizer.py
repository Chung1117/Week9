import streamlit as st
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def main():
    st.title("AI-Powered Text Summarizer")
    st.subheader("Summarize long articles into concise summaries with AI")
    
    # User input for the text to be summarized
    user_input_text = st.text_area("Enter the text you want to summarize:", height=300)
    
    # Button to trigger summarization
    if st.button("Summarize"):
        if user_input_text:
            # Generating the summary
            summary = summarizer(user_input_text, max_length=130, min_length=30, do_sample=False)
            st.write("Summary:")
            st.write(summary[0]['summary_text'])
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()