
import streamlit as st
import html

from scraper.wiki_scraper import scrape_wikipedia_pages
from embedder.chunk_and_embed import chunk_texts, embed_chunks
from indexer.faiss_index import create_faiss_index, search_faiss_index
from qa_model.answer_question import get_answer


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Talash+ | AI Research Assistant",
    page_icon="🔮",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #080d1c, #11162b, #17132f);
    color: #f5f5ff;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main headings */

h1, h2, h3, p, label {
    color: #f5f5ff !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #10162b !important;
    border-right: 1px solid #303957;
}

section[data-testid="stSidebar"] * {
    color: #f5f5ff !important;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #b99aff !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #343d60 !important;
}

/* Hero section */

.hero {
    background: linear-gradient(120deg, #25204a, #171d3b);
    padding: 38px;
    border-radius: 22px;
    border: 1px solid #51427e;
    margin-bottom: 30px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: #d0c8ed;
    font-size: 16px;
    line-height: 1.8;
}

/* Section headings */

.small-heading {
    color: #b9a5ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-top: 18px;
}

/* Input boxes */

.stTextArea textarea,
.stTextInput input {
    background-color: #10162b !important;
    color: #ffffff !important;
    border: 1px solid #514b79 !important;
    border-radius: 12px !important;
}

.stTextArea textarea::placeholder,
.stTextInput input::placeholder {
    color: #929bb5 !important;
}

/* Search button */

.stButton button {
    background: linear-gradient(90deg, #7852d6, #a45be8);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 13px 25px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton button:hover {
    background: linear-gradient(90deg, #9068ed, #bd70f5);
    border: none;
    transform: translateY(-2px);
}

/* Answer card */

.answer-card {
    background: linear-gradient(120deg, #201d43, #191d38);
    border: 1px solid #8062ce;
    border-radius: 18px;
    padding: 26px;
    margin-top: 15px;
}

.answer-label {
    color: #b99aff;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 15px;
}

.answer-text {
    color: #ffffff;
    font-size: 19px;
    line-height: 1.8;
}

/* Source passages */

div[data-testid="stExpander"] {
    background: #171d35;
    border: 1px solid #343d5c;
    border-radius: 12px;
}

/* Metrics */

div[data-testid="stMetric"] {
    background: #1b2240;
    border: 1px solid #343d60;
    padding: 18px;
    border-radius: 14px;
}

/* Footer */

.footer {
    color: #929bb5;
    text-align: center;
    font-size: 13px;
    padding: 30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.markdown("## 🔮 Talash+")

    st.caption("AI Research Assistant")

    st.markdown("---")

    st.markdown("### Navigation")

    st.markdown("🔍 Wikipedia Search")
    st.markdown("🧠 AI Question Answering")
    st.markdown("📚 Source Passages")

    st.markdown("---")

    st.markdown("### Technology")

    st.markdown("🐍 Python")
    st.markdown("🤗 Sentence Transformers")
    st.markdown("⚡ FAISS")
    st.markdown("✨ FLAN-T5-base")
    st.markdown("🎨 Streamlit")

    st.markdown("---")

    st.caption("Talash+ | Knowledge Retrieval System")


# =====================================
# HERO SECTION
# =====================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🔮 Talash+</div>
<div class="hero-subtitle">
Your intelligent Wikipedia research assistant.
Search knowledge, discover information and get
AI-powered answers from Wikipedia.
</div>
</div>
""",
    unsafe_allow_html=True
)


# =====================================
# SEARCH SECTION
# =====================================

st.markdown("## 🔍 Ask Your Question")

st.markdown(
    '<div class="small-heading">STEP 01 — ADD YOUR KNOWLEDGE SOURCE</div>',
    unsafe_allow_html=True
)

urls_input = st.text_area(
    "Wikipedia URLs",
    placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
    height=110
)

st.markdown(
    '<div class="small-heading">STEP 02 — ENTER YOUR QUESTION</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Your question",
    placeholder="What is artificial intelligence?"
)

st.write("")

search_button = st.button(
    "✨  Find Answer",
    type="primary",
    use_container_width=True
)


# =====================================
# QUESTION ANSWERING PIPELINE
# =====================================

if search_button:

    if not urls_input.strip() or not question.strip():

        st.warning(
            "Please enter a Wikipedia URL and your question."
        )

    else:

        try:

            # STEP 1: Wikipedia scraping

            with st.spinner("🌐 Reading Wikipedia pages..."):

                pages = scrape_wikipedia_pages(urls_input)

            if not pages:

                st.error(
                    "No Wikipedia content found. Please check your URLs."
                )

            else:

                # STEP 2: Text chunking

                with st.spinner("📚 Preparing text chunks..."):

                    chunks = chunk_texts(pages)

                if not chunks:

                    st.error("No usable text was found.")

                else:

                    # STEP 3: Embeddings

                    with st.spinner("🧠 Creating text embeddings..."):

                        embeddings = embed_chunks(chunks)

                    # STEP 4: FAISS indexing

                    index = create_faiss_index(embeddings)

                    # STEP 5: Semantic search

                    with st.spinner("🔎 Searching relevant information..."):

                        relevant_chunks = search_faiss_index(
                            index,
                            question,
                            chunks,
                            top_k=4
                        )

                    if not relevant_chunks:

                        st.warning("No relevant information found.")

                    else:

                        # STEP 6: Generate answer

                        with st.spinner("✨ Generating your answer..."):

                            answer = get_answer(
                                question,
                                relevant_chunks
                            )

                        # =====================================
                        # ANSWER DISPLAY
                        # =====================================

                        st.markdown("---")

                        st.markdown("## 🧠 AI Generated Answer")

                        if answer:

                            safe_answer = html.escape(str(answer))

                            st.markdown(
                                f"""
<div class="answer-card">
<div class="answer-label">
TALASH+ ANSWER
</div>
<div class="answer-text">
{safe_answer}
</div>
</div>
""",
                                unsafe_allow_html=True
                            )

                        else:

                            st.info(
                                "No confident answer found in the selected pages."
                            )

                        # =====================================
                        # PROJECT STATISTICS
                        # =====================================

                        st.markdown("### 📊 Search Statistics")

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Pages Retrieved",
                                len(pages)
                            )

                        with col2:

                            st.metric(
                                "Text Chunks",
                                len(chunks)
                            )

                        with col3:

                            st.metric(
                                "Relevant Passages",
                                len(relevant_chunks)
                            )

                        # =====================================
                        # SOURCE PASSAGES
                        # =====================================

                        st.markdown("### 📚 Retrieved Sources")

                        with st.expander(
                            "View Wikipedia source passages",
                            expanded=False
                        ):

                            for i, chunk in enumerate(
                                relevant_chunks,
                                start=1
                            ):

                                st.markdown(f"**Passage {i}**")

                                st.write(chunk)

                                st.markdown("---")

                        st.caption(
                            "Answers are based on retrieved Wikipedia text. "
                            "Verify important information against the original sources."
                        )

        except Exception as exc:

            st.error(f"Something went wrong: {exc}")

            st.info(
                "Check your Python modules and requirements. "
                "The first model download may take some time."
            )


# =====================================
# FOOTER
# =====================================

st.markdown(
    """
<div class="footer">
Built with Python, Streamlit, FAISS and Hugging Face.
<br><br>
© Talash+ | AI-Powered Knowledge Retrieval System
</div>
""",
    unsafe_allow_html=True
)
