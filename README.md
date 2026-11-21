# Talash+ | AI-Powered Wikipedia Question Answering System

Talash+ is an AI-powered Question Answering (QA) system that allows users to ask questions about Wikipedia articles and receive relevant answers. It uses Natural Language Processing (NLP), transformer-based models, text embeddings, and FAISS similarity search to retrieve relevant information and generate answers.

## About the Project

The project is designed to make information retrieval easier by allowing users to provide Wikipedia article URLs and ask questions based on their content.

Talash+ processes the article text, divides it into smaller chunks, converts the chunks into numerical embeddings, and uses FAISS to retrieve relevant information. A transformer-based Question Answering model then extracts an answer from the retrieved content.

## Features

- Accepts multiple English Wikipedia article URLs.
- Extracts article content from Wikipedia.
- Splits text into smaller, overlapping chunks.
- Generates text embeddings using Sentence Transformers.
- Uses FAISS for similarity-based information retrieval.
- Extracts answers using a transformer-based Question Answering model.
- Provides a simple interface using Streamlit.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| Natural Language Processing (NLP) | Text processing |
| Sentence Transformers | Text embeddings |
| FAISS | Similarity search |
| Hugging Face Transformers | Question Answering model |
| Wikipedia API | Article content retrieval |

## AI Models

**Embedding Model:**  
`sentence-transformers/all-MiniLM-L6-v2`

**Question Answering Model:**  
`deepset/roberta-base-squad2`

## Project Architecture

```text
TalashPlus/
│
├── app.py
├── README.md
├── requirements.txt
│
├── scraper/
│   └── wiki_scraper.py
│
├── embedder/
│   └── chunk_and_embed.py
│
├── indexer/
│   └── faiss_index.py
│
└── qa_model/
    └── answer_question.py
```

## How It Works

1. **Article Extraction:** The system fetches article content from the provided Wikipedia URLs.
2. **Text Chunking:** The extracted content is divided into smaller, overlapping text chunks.
3. **Text Embedding:** Sentence Transformers converts the text chunks into numerical embeddings.
4. **Information Retrieval:** FAISS searches for chunks relevant to the user's question.
5. **Answer Extraction:** The Question Answering model extracts an answer from the retrieved content.
6. **User Interface:** Streamlit displays the results to the user.

## Installation and Setup

### Prerequisites

- Python 3.10 or 3.11
- Git
- Internet connection

### 1. Clone the Repository

```bash
git clone https://github.com/Ambika0121/Talash-Plus-Wikipedia-QA.git
```

### 2. Navigate to the Project Folder

```bash
cd Talash-Plus-Wikipedia-QA/TalashPlus_Project/TalashPlus
```

### 3. Create a Virtual Environment

```bash
py -3.11 -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## Example Usage

**Wikipedia Article URL:**

`https://en.wikipedia.org/wiki/Artificial_intelligence`

**Sample Question:**

What is artificial intelligence?

The system retrieves relevant information from the article and displays an answer.

## Future Enhancements

- Support for PDF document question answering.
- Integration with additional knowledge sources.
- Improvements to information retrieval and answer quality.
- Enhancements to the user interface.
- Evaluation of answer relevance and accuracy.

## Learning Outcomes

Through this project, I gained practical exposure to:

- Python programming.
- Natural Language Processing concepts.
- Transformer-based models.
- Text embeddings and vector search.
- FAISS similarity search.
- Building applications using Streamlit.

## Author

**Ambika Patil**  
MCA Graduate | AI and Generative AI Enthusiast

GitHub: https://github.com/Ambika0121
