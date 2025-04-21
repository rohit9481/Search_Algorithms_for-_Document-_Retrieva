import re
import pandas as pd
import nltk
import difflib
from nltk.corpus import wordnet
from collections import defaultdict
from io import StringIO
import streamlit as st

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def correct_spelling(query, vocab):
    corrected = []
    for word in query.split():
        if word in vocab:
            corrected.append(word)
        else:
            close = difflib.get_close_matches(word, vocab, n=1)
            corrected.append(close[0] if close else word)
    return " ".join(corrected)

def expand_query_with_synonyms(query):
    words = query.split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)

def highlight_keywords(text, keywords):
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(lambda m: f"**:orange[{m.group(0)}]**", text)
    return text

def export_results(data):
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    txt = "\n\n".join([f"Title: {row['Title']}\nFile: {row['File']}\nScore: {row['Score']}\nPreview:\n{row['Preview']}" for _, row in df.iterrows()])

    st.download_button("📥 Download as TXT", data=txt, file_name="results.txt", mime="text/plain")
