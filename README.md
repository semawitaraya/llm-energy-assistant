# LLM Energy Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions 
grounded in Dutch energy sector  annual reports.

Built as part of my EngD in Data Science at JADS.

## What it does

Ask natural language questions about Dutch energy sector reports:

> "What are Alliander's CO₂ targets for 2030?"  
> "How much did Enexis invest in the grid in 2023?"  
> "What is grid congestion and why is it a problem?"

The app retrieves the most relevant passages from the documents 
and uses an LLM to write a grounded answer with source references.

## Architecture

PDF documents → chunking → embeddings → FAISS vector store
↓
User question → embed → similarity search → top 4 chunks
↓
LLM generates grounded answer

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| LLM | OpenAI GPT-3.5-turbo / HuggingFace fallback |
| Deployment | Render |

##  Run locally

```bash
# 1. Clone the repo
git clone https://github.com/[your-username]/llm-energy-assistant.git
cd llm-energy-assistant

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key (optional)
cp .env.example .env
# edit .env and add your OPENAI_API_KEY

# 5. Build the index
python ingest.py

# 6. Run the app
streamlit run app.py
```

## Project structure

```
llm-energy-assistant/
├── src/
│   ├── config.py            # centralised settings
│   ├── document_loader.py   # PDF loading and chunking
│   └── rag_pipeline.py      # embeddings, FAISS, LLM chain
├── data/                    # drop PDFs here
├── app.py                   # Streamlit frontend
├── ingest.py                # builds the FAISS index
└── requirements.txt
## License
MIT
