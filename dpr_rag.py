import os
import uuid
import shutil
import time
from typing import List, Generator, Tuple
import numpy as np

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb

# Directory setup
UPLOAD_DIR = "uploaded_pdfs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False
if "client" not in st.session_state:
    st.session_state.client = None
if "uploader_counter" not in st.session_state:
    st.session_state.uploader_counter = 0

# Initialize components
query_encoder = OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=1)
passage_encoder = OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=1)
llm = OllamaLLM(model="qwen2.5:latest", temperature=0.3, num_gpu=1)

class DPRRetriever:
    def __init__(self, vectorstore, query_encoder):
        self.vectorstore = vectorstore
        self.query_encoder = query_encoder
        
    def retrieve(self, query: str, retrieve_k: int = 8, return_k: int = 3) -> Tuple[List[Document], np.ndarray]:
        # Encode query with query encoder
        query_embedding = self.query_encoder.embed_query(query)
        
        # Retrieve more documents initially (retrieve_k)
        docs_and_scores = self.vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=retrieve_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process all retrieved results
        all_docs = []
        all_scores = []
        for i in range(len(docs_and_scores["ids"][0])):
            doc = Document(
                page_content=docs_and_scores["documents"][0][i],
                metadata=docs_and_scores["metadatas"][0][i]
            )
            all_docs.append(doc)
            all_scores.append(1.0 - docs_and_scores["distances"][0][i])  # Convert distance to similarity
        
        all_scores = np.array(all_scores)
        
        # Normalize scores with softmax
        norm_scores = np.exp(all_scores) / np.sum(np.exp(all_scores))
        
        # Sort documents by normalized scores in descending order
        sorted_indices = np.argsort(norm_scores)[::-1]
        sorted_docs = [all_docs[i] for i in sorted_indices]
        sorted_scores = norm_scores[sorted_indices]
        
        # Return only the top return_k documents
        return sorted_docs[:return_k], sorted_scores[:return_k]

class DPRChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def run(self, question: str) -> Generator[str, None, None]:
        docs, scores = self.retriever.retrieve(question)
        dpr_prompt = self._build_dpr_prompt(question, docs, scores)
        
        # Create a generator that yields the response in chunks
        response = self.llm.stream(dpr_prompt)
        for chunk in response:
            yield chunk
    
    def _build_dpr_prompt(self, question: str, docs: List[Document], scores: np.ndarray) -> str:
        prompt = """You are an expert AI assistant with multi-source analysis capabilities. Using the following documents:\n\n"""
        
        for i, (doc, score) in enumerate(zip(docs, scores)):
            prompt += f"""--- Document {i+1} (Confidence: {score:.2f}) ---\n{doc.page_content}\n\n"""
        
        prompt += f"""\nPlease answer this question:\n{question}\n\nYour response should:\n"""
        prompt += """1. Summarize relevant information from each document\n"""
        prompt += """2. Synthesize information for comprehensive answer\n"""
        prompt += """3. Mention relevant standards\n"""
        prompt += """4. Provide suggested practical solutions\n"""
        
        return prompt

def process_uploaded_pdfs(uploaded_files):
    documents = []
    
    for uploaded_file in uploaded_files:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            for chunk in chunks:
                chunk.metadata['source'] = uploaded_file.name
            documents.extend(chunks)
        except Exception as e:
            os.remove(file_path)
            raise e
    
    if not documents:
        raise ValueError("No valid documents were processed")
    
    # Initialize Chroma client with explicit settings
    st.session_state.client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete any existing collection to ensure clean start
    try:
        st.session_state.client.delete_collection("pdf_collection")
    except:
        pass
    
    # Create vector store with explicit client and passage encoder
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=passage_encoder,
        client=st.session_state.client,
        collection_name="pdf_collection"
    )
    
    return vectorstore

def clear_documents():
    try:
        # Reset session state first to release resources
        st.session_state.vectorstore = None
        st.session_state.documents_processed = False
        st.session_state.messages = []
        st.session_state.clear_flag = True
        
        # Explicitly delete the Chroma collection if it exists
        if st.session_state.client:
            try:
                st.session_state.client.delete_collection("pdf_collection")
            except Exception as e:
                st.warning(f"Warning: Could not delete collection - {str(e)}")
        
        # Close the Chroma client
        if st.session_state.client:
            try:
                st.session_state.client = None
            except:
                pass
        
        # Give time for resources to be released
        time.sleep(2)
        
        # Remove all uploaded PDFs
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Remove Chroma database with retries and force deletion
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.path.exists(CHROMA_DIR):
                    # On Windows, we need to handle file locking explicitly
                    if os.name == 'nt':
                        os.system(f'rmdir /s /q "{CHROMA_DIR}"')
                    else:
                        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to clear ChromaDB after {max_retries} attempts: {str(e)}")
                    break
                time.sleep(2)
        
        # Increment the uploader key to reset the file uploader
        st.session_state.uploader_counter += 1
        st.session_state.clear_flag = False
        
        st.success("All documents have been completely cleared. You can upload new documents.")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing documents: {str(e)}")

def main():
    st.set_page_config(page_title="PDF Q&A with DPR RAG", page_icon="📚")
    st.title("📚 PDF Q&A with Dense Passage Retrieval RAG")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File uploader with unique key that changes when cleared
        uploader_key = f"file_uploader_{st.session_state.uploader_counter}"
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key=uploader_key
        )
        
        # Process documents button
        if st.button("Process Documents") and uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    vectorstore = process_uploaded_pdfs(uploaded_files)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = DPRRetriever(vectorstore, query_encoder)
                    st.session_state.chain = DPRChain(st.session_state.retriever, llm)
                    
                    st.session_state.documents_processed = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I'm ready to answer questions about your documents using DPR RAG!"
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        
        # Clear documents button
        if st.button("Clear All Documents"):
            clear_documents()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if not st.session_state.documents_processed:
            with st.chat_message("assistant"):
                st.warning("Please upload and process PDF documents first")
        else:
            with st.chat_message("assistant"):
                response = st.write_stream(st.session_state.chain.run(prompt))
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()