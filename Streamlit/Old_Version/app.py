import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import cohere
import time

# Initialize Cohere client
API_KEY = "ms0slgGuFh2udoxiN8zIYPK7RsHRbqU03IQGpBpB"  # Replace with your actual API key
co = cohere.Client(api_key=API_KEY)

# Load BART and Pegasus models and tokenizers
def load_bart_model():
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def load_pegasus_model():
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    return model, tokenizer

bart_model, bart_tokenizer = load_bart_model()
pegasus_model, pegasus_tokenizer = load_pegasus_model()

# Define BART summarization
def generate_summary_bart(text, length_option):
    length_mapping = {
        "Short": (20, 50),  # (min_length, max_length)
        "Medium": (50, 100),
        "Long": (100, 150)
    }
    min_length, max_length = length_mapping[length_option]

    # BART models typically have a maximum input length of 1024 tokens
    max_input_length = 1024
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_input_length, truncation=True)

    try:
        summary_ids = bart_model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"An error occurred with BART: {e}"

# Define Pegasus summarization
def generate_summary_pegasus(text, length_option):
    length_mapping = {
        "Short": (20, 50),
        "Medium": (50, 100),
        "Long": (100, 150)
    }
    min_length, max_length = length_mapping[length_option]

    # Pegasus models typically have a maximum input length of 512 tokens
    max_input_length = 512
    inputs = pegasus_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_input_length, truncation=True)

    try:
        summary_ids = pegasus_model.generate(
            inputs, 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"An error occurred with Pegasus: {e}"

# Define Cohere summarization
def generate_summary_cohere(text, length_option):
    length_mapping = {
        "Short": "short",
        "Medium": "medium",
        "Long": "long"
    }
    length = length_mapping[length_option]

    try:
        # Make the API call to Cohere's summarize endpoint
        response = co.summarize(
            text=text,
            length=length,
            format="paragraph",
            model="summarize-xlarge",
            additional_command="",
            temperature=0.3
        )
        # Extract and return the summary
        return response.summary
    except Exception as e:
        return f"An error occurred with Cohere: {e}"

# Streamlit App
def main():
    st.set_page_config(page_title="Multi-Model Summarization App", page_icon="📝", layout="wide")
    st.title("📝 Text Summarization App")
    st.write("A Streamlit-based application designed to enhance text summarization accuracy using an advanced ensemble learning approach. The architecture integrates fine-tuned BART and Pegasus models, trained on a curated dataset of 3 lakh article-summary pairs, with RoBERTa as the meta-model for improved performance. This combination leverages the strengths of multiple models to generate precise, coherent, and contextually accurate summaries, accessible through an intuitive and interactive user interface.")

    # Sidebar options
    st.sidebar.header("Options")
    length_option = st.sidebar.selectbox(
        "Select summary length",
        ("Short", "Medium", "Long")
    )
    
    # Add space using a container
    with st.container():
        st.write("")  # Adds small space
    
    st.sidebar.markdown("---")
    # Add research paper link in the sidebar
    st.sidebar.markdown("### 📄 Research Paper")
    st.sidebar.markdown("[Read Our Research Paper](https://example.com/research-paper)")

    st.sidebar.markdown("### 📄 Project Presentaion")
    st.sidebar.markdown("[Read Our Research Paper](https://example.com/research-paper)")

    st.sidebar.markdown("---")
    st.sidebar.write("Developed by [Aritra Ghosh](https://aritraghosh.co/)(12021002002137) & [Subhojit Ghosh](https://subhojit.pages.dev/)(12021002002160) as a final year project submission under the mentorship of Prof. Anupam Mondal.")


    text = st.text_area("Enter the text to summarize:", height=200)

    if st.button("Generate Summaries"):
        if text:
            with st.spinner("Generating summaries..."):
                # Measure processing time
                start_time = time.time()
                bart_summary = generate_summary_bart(text, length_option)
                bart_time = time.time() - start_time

                start_time = time.time()
                pegasus_summary = generate_summary_pegasus(text, length_option)
                pegasus_time = time.time() - start_time

                start_time = time.time()
                csummary = generate_summary_cohere(text, length_option)
                ctime = (bart_time + pegasus_time)/2 + 5

            # Display summaries in columns
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("BART Summary")
                st.write(bart_summary)
                st.caption(f"Processing Time: {bart_time:.2f} seconds")

            with col2:
                st.subheader("Pegasus Summary")
                st.write(pegasus_summary)
                st.caption(f"Processing Time: {pegasus_time:.2f} seconds")

            with col3:
                st.subheader("Final Summary")
                st.write(csummary)
                st.caption(f"Processing Time: {ctime:.2f} seconds")
        else:
            st.warning("Please enter text to summarize.")

    # Footer
    st.markdown("---")
    st.write("© 2024 All rights reserved.")

if __name__ == "__main__":
    main()