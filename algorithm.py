# Import necessary libraries
import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import torch

# Download NLTK data for sentence tokenization
nltk.download('punkt')

# Load BART and Pegasus models and tokenizers
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

# Load RoBERTa model for semantic similarity
roberta_model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

# Define BART summary function
def generate_summary_bart(text):
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    bart_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return bart_summary

# Define Pegasus summary function
def generate_summary_pegasus(text):
    inputs = pegasus_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = pegasus_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    pegasus_summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return pegasus_summary

# Function to combine summaries based on relevance to the original text
def combine_summaries(text, bart_summary, pegasus_summary):
    bart_sentences = sent_tokenize(bart_summary)
    pegasus_sentences = sent_tokenize(pegasus_summary)
    text_embedding = roberta_model.encode(text, convert_to_tensor=True)

    final_summary_sentences = []

    for sentence in bart_sentences:
        sentence_embedding = roberta_model.encode(sentence, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(sentence_embedding, text_embedding)[0].item()
        if similarity > 0.6:
            final_summary_sentences.append(sentence)

    for sentence in pegasus_sentences:
        if sentence not in final_summary_sentences:
            sentence_embedding = roberta_model.encode(sentence, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(sentence_embedding, text_embedding)[0].item()
            if similarity > 0.6:
                final_summary_sentences.append(sentence)

    final_summary = ' '.join(final_summary_sentences)
    return final_summary

# Streamlit App
def main():
    st.title("Ensemble Abstractive Summarization App")
    st.write("This app uses BART and Pegasus models to generate summaries and refines them with RoBERTa for semantic relevance.")

    text = st.text_area("Enter the text to summarize", height=300)
    
    if st.button("Generate Summary"):
        if text:
            with st.spinner("Generating summaries..."):
                bart_summary = generate_summary_bart(text)
                pegasus_summary = generate_summary_pegasus(text)
                
                final_summary = combine_summaries(text, bart_summary, pegasus_summary)
                
                st.subheader("BART Summary")
                st.write(bart_summary)
                
                st.subheader("Pegasus Summary")
                st.write(pegasus_summary)
                
                st.subheader("Final Ensemble Summary")
                st.write(final_summary)
        else:
            st.warning("Please enter text to summarize.")

if __name__ == "__main__":
    main()
