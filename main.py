import streamlit as st 
from datetime import datetime
from search_engine import DocumentSearchEngine
from utils import (
    preprocess_text,
    expand_query_with_synonyms,
    highlight_keywords,
    correct_spelling,
    export_results
)
from analytics import update_history, get_history_df

# Initialize state
if "engine" not in st.session_state:
    st.session_state.engine = DocumentSearchEngine()
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Sidebar navigation
st.sidebar.title("📘 Navigation")
page = st.sidebar.radio("Go to", [
    "Upload & Manage Documents",
    "Search Engine",
    "Export Results",
    "Search Analytics"
])

st.title("🔍 Document Retrieval System")

# --- Upload & Manage Documents ---
if page == "Upload & Manage Documents":
    st.header("📂 Upload & Manage Documents")
    uploaded_files = st.file_uploader("Upload .txt files", type="txt", accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            content = file.read().decode("utf-8")
            file_data = {
                "filename": file.name,
                "title": file.name.replace(".txt", "").title(),
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": content
            }
            st.session_state.uploaded_files.append(file_data)
        st.success("Files uploaded!")

    if st.button("🔄 Rebuild Index"):
        st.session_state.engine.build_index(st.session_state.uploaded_files)
        st.success("Index rebuilt successfully!")

    if st.session_state.uploaded_files:
        st.subheader("📑 Uploaded Documents")
        for doc in st.session_state.uploaded_files:
            st.markdown(f"• `{doc['title']}` | 📁 `{doc['filename']}` | 🕓 {doc['uploaded_at']}")

# --- Search Engine ---
elif page == "Search Engine":
    st.header("🔎 Search Engine")
    all_metadata = st.session_state.engine.get_metadata()
    titles = list({meta["title"] for meta in all_metadata})
    selected_titles = st.multiselect("Filter by Document Title", titles)

    query = st.text_input("Enter your query")
    expand = st.checkbox("Expand Query with Synonyms")
    correct = st.checkbox("Correct Spelling")

    top_k = st.slider("Top K Results", 1, 20, 5)
    alpha = st.slider("Hybrid Weight (TF-IDF vs BM25)", 0.0, 1.0, 0.5)

    if st.button("Search") and query:
        # Filter docs based on selected titles
        filtered_docs = st.session_state.uploaded_files
        if selected_titles:
            filtered_docs = [doc for doc in filtered_docs if doc["title"] in selected_titles]

        st.session_state.engine.build_index(filtered_docs)

        original_query = query

        if correct:
            vocab = st.session_state.engine.get_vocabulary()
            query = correct_spelling(query, vocab)

        if expand:
            query_terms = expand_query_with_synonyms(query)
            query = " ".join(query_terms)

        update_history(st.session_state.search_history, original_query)
        st.session_state.last_query = original_query

        results = st.session_state.engine.hybrid_search(query, top_n=top_k, alpha=alpha)
        st.session_state.search_results = results

        if results:
            st.subheader("🔍 Top Results")
            for idx, score in results:
                doc = st.session_state.uploaded_files[idx]
                meta = st.session_state.engine.metadata[idx]
                preview = highlight_keywords(doc["content"][:500], query.split())

                st.markdown(f"""
                **📄 Title**: {meta['title']}  
                **📁 File**: {meta['filename']}  
                **🕓 Uploaded**: {meta['uploaded_at']}  
                **🧾 Length**: {meta['length']} words  
                **⭐ Score**: `{round(score, 4)}`  
                ---
                {preview}...
                """)
        else:
            st.warning("No matching results found.")

# --- Export Results ---
elif page == "Export Results":
    st.header("📤 Export Search Results")
    if not st.session_state.search_results:
        st.warning("No results to export. Run a search first.")
    else:
        export_data = []
        for idx, score in st.session_state.search_results:
            doc = st.session_state.uploaded_files[idx]
            meta = st.session_state.engine.metadata[idx]
            export_data.append({
                "Title": meta['title'],
                "File": meta['filename'],
                "Uploaded": meta['uploaded_at'],
                "Score": round(score, 4),
                "Preview": doc["content"][:300]
            })
        export_results(export_data)

# --- Search Analytics ---
elif page == "Search Analytics":
    st.header("📊 Search Analytics")
    df = get_history_df(st.session_state.search_history)
    if df.empty:
        st.info("No search history yet.")
    else:
        st.dataframe(df)
        st.bar_chart(df.set_index("Query")["Frequency"])
