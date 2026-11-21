# Talash+ — AI-Powered Wikipedia Question Answering System

A modular Python starter project that retrieves relevant passages from English Wikipedia pages and extracts an answer using a question-answering model.

## Features
- Enter one or more English Wikipedia article URLs.
- Fetch article text using `wikipedia-api`.
- Split text into overlapping chunks.
- Create semantic embeddings with `all-MiniLM-L6-v2`.
- Retrieve relevant chunks using FAISS.
- Extract answers using `deepset/roberta-base-squad2`.
- Display results in Streamlit.

## Project structure
```text
TalashPlus/
├── app.py
├── requirements.txt
├── README.md
├── scraper/
├── embedder/
├── indexer/
└── qa_model/
```

## Setup (Windows, Python 3.10 or 3.11 recommended)
1. Extract this ZIP.
2. Open the `TalashPlus` folder in VS Code.
3. Open Terminal and run:

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local address, usually `http://localhost:8501`. Open it in your browser.

## Example
Wikipedia URL:
`https://en.wikipedia.org/wiki/Artificial_intelligence`

Question:
`What is artificial intelligence?`

## Notes
- An internet connection is needed for Wikipedia and the first-time model downloads.
- The first run can take several minutes as the MiniLM and RoBERTa model files download.
- This is a clean starter implementation based on the project idea and main app file shared in the chat. It is not a recovery of the missing original modules.
- This version uses MiniLM embeddings, FAISS, and RoBERTa-base-SQuAD2. It does not currently use LangChain or LLaMA 3.
- The claimed 41% speed improvement and 31% accuracy improvement have not been independently tested. Do not put those metrics on a resume unless you conduct and document a valid evaluation.
- Test the application and understand each module before listing it as completed on your resume.
