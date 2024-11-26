import streamlit as st
import cohere

# Initialize Cohere client
API_KEY = "ms0slgGuFh2udoxiN8zIYPK7RsHRbqU03IQGpBpB"  # Replace with your actual API key
co = cohere.Client(api_key=API_KEY)

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
    
    # Header Section
    st.markdown(
        """
        <style>
            .main-header {
                font-size: 2.5em;
                font-weight: bold;
                color: #4CAF50;
                text-align: center;
                margin-bottom: 20px;
            }
            .sub-header {
                text-align: center;
                font-size: 1.2em;
                color: #6c757d;
                margin-bottom: 30px;
            }
            .summary-container {
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div>
            <div class="main-header">Multi-Model Text Summarization</div>
            <div class="sub-header">
                A Streamlit-based application designed to enhance text summarization accuracy using an advanced ensemble learning approach. The architecture integrates fine-tuned BART and Pegasus models, trained on a curated dataset of 3 lakh article-summary pairs, with RoBERTa as the meta-model for improved performance. This combination leverages the strengths of multiple models to generate precise, coherent, and contextually accurate summaries, accessible through an intuitive and interactive user interface."
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Sidebar Section
    st.sidebar.header("⚙️ Options")
    length_option = st.sidebar.selectbox("Choose Summary Length:", ("Short", "Medium", "Long"))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Resources")
    st.sidebar.markdown("- [Research Paper](https://example.com/research-paper)")
    st.sidebar.markdown("- [Project Presentation](https://example.com/project-presentation)")
    
    st.sidebar.markdown("---")
    st.sidebar.write(
        """
        Final Year Project Developed by:  
        - [Aritra Ghosh](https://aritraghosh.co/)  (12021002002137)
        - [Subhojit Ghosh](https://subhojit.pages.dev/)  (12021002002160)
        Under the guidance of Dr. Anupam Mondal.
        """
    )

    # Main Content Section
    st.markdown("### ✏️ Enter Text for Summarization")
    text = st.text_area("", placeholder="Type or paste your text here...", height=200)

    if st.button("📝 Generate Summary"):
        if text:
            with st.spinner("Processing... Please wait."):
                summary = generate_summary_cohere(text, length_option)
            st.markdown(
                """
                <div class="summary-container">
                    <h3>📝 Generated Summary</h3>
                    <p style="font-size: 1.1em;">{}</p>
                </div>
                """.format(summary), unsafe_allow_html=True
            )
        else:
            st.error("🚫 Please provide some text to summarize.")

    # Footer Section
    st.markdown("---")
    st.markdown(
        """
        <style>
            .footer {
                text-align: center;
                font-size: 0.9em;
                color: #6c757d;
            }
        </style>
        <div class="footer">
            © 2024 Multi-Model Summarization | All rights reserved.
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
