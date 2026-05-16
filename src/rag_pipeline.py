import logging
from pathlib import Path
from typing import Tuple

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import settings
from src.document_loader import chunk_documents, load_documents

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are an expert energy sector analyst.
Use ONLY the context passages below to answer the question.
If the answer is not in the context, say "I don't have enough
information in the loaded reports to answer that."

Be precise, cite specific figures or years when available,
and keep your answer concise.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

FAISS_INDEX_PATH = settings.vectorstore_dir / "faiss_index"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_or_create_vectorstore() -> Tuple[FAISS, int]:
    """Load FAISS index from disk, or build it if it doesn't exist."""
    embeddings = get_embeddings()
    index_file = FAISS_INDEX_PATH / "index.faiss"

    if index_file.exists():
        logger.info("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        n_docs = vectorstore.index.ntotal
        logger.info("Loaded %d vectors", n_docs)
        return vectorstore, n_docs

    logger.info("No index found — building from scratch...")
    return _build_vectorstore(embeddings)


def _build_vectorstore(embeddings: HuggingFaceEmbeddings) -> Tuple[FAISS, int]:
    """Embed all document chunks and save FAISS index to disk."""
    docs = load_documents()
    chunks = chunk_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_PATH))

    logger.info("Built and saved index with %d chunks", len(chunks))
    return vectorstore, len(chunks)

def _get_llm():
    """Return OpenAI LLM if key is set, otherwise HuggingFace fallback."""
    if settings.use_openai:
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI: %s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=512,
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_community.llms import HuggingFacePipeline

    model_id = "facebook/opt-125m"
    logger.info("Using HuggingFace fallback: %s", model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=pipe)


def build_rag_chain(vectorstore: FAISS, top_k: int = 4) -> RetrievalQA:
    """Connect retriever + LLM into one chain."""
    llm = _get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    logger.info("RAG chain ready (top_k=%d)", top_k)
    return chain