import spacy
from transformers import AutoTokenizer
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
text = "This is an example text for summarization preprocessing."
doc = nlp(text)
lemmatized_text = " ".join([token.lemma_ for token in doc])
tokens = tokenizer.tokenize(lemmatized_text)
print("Tokens:", tokens)


