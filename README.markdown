# DPR RAG: A Streamlit-Powered PDF Q&A System with Dense Passage Retrieval

[![GitHub Stars](https://img.shields.io/github/stars/armanjscript/DPR-RAG?style=social)](https://github.com/armanjscript/DPR-RAG)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Description

Welcome to **DPR RAG**, a cutting-edge web-based application designed to answer questions based on the content of uploaded PDF documents. Leveraging **Dense Passage Retrieval (DPR)** for **Retrieval-Augmented Generation (RAG)**, this project combines semantic similarity with advanced retrieval techniques to deliver precise and contextually relevant responses. Built with modern technologies such as **Streamlit**, **LangChain**, **Chroma**, and **Ollama**, DPR RAG is an ideal tool for researchers, students, or professionals seeking to extract insights from documents efficiently.

The application offers a user-friendly interface where users can upload PDFs, ask questions in a conversational format, and receive detailed, real-time responses with citations to source documents, complete with confidence scores for transparency and verifiability. It seems likely that this tool is particularly valuable for tasks like academic research, legal document analysis, or technical documentation review, where quick and accurate information retrieval is essential.

## Features

| Feature | Description |
|---------|-------------|
| **PDF Upload & Processing** | Upload multiple PDF files, which are automatically split into chunks (1000 characters, 200-character overlap) and indexed for querying. |
| **Dense Passage Retrieval** | Employs DPR to retrieve up to 8 relevant document passages based on semantic similarity, selecting the top 3 for response generation. |
| **Conversational Interface** | Engage in a chat-like interface to ask questions and receive detailed answers based on document content. |
| **Real-Time Responses** | Answers are streamed in real-time, ensuring a seamless user experience. |
| **Error Handling** | Robust mechanisms handle errors and allow cleanup of uploaded files and vector stores. |
| **Source Citations** | Responses include references to source documents with normalized confidence scores, enhancing trust and traceability. |

## How It Works

The DPR RAG system operates through a sophisticated pipeline that processes documents, retrieves relevant passages, and generates answers. Below is a detailed breakdown of the process:

### Document Processing
- **PDF Loading**: PDFs are loaded using `PyPDFLoader` from LangChain, which extracts text from uploaded files.
- **Text Splitting**: Documents are divided into chunks of 1000 characters with a 200-character overlap using `RecursiveCharacterTextSplitter` to maintain context.
- **Metadata Tagging**: Each chunk is tagged with metadata, such as the source file name, to ensure traceability in responses.

### Retrieval
The retrieval process utilizes **Dense Passage Retrieval (DPR)** to identify the most relevant document chunks:
- **Embedding Generation**: Both documents and queries are embedded into a high-dimensional vector space using `OllamaEmbeddings` (model: `nomic-embed-text:latest`) with GPU support (`num_gpu=1`).
- **Vector Storage**: Embeddings are stored in a Chroma vector store, powered by ChromaDB, for efficient similarity searches.
- **DPRRetriever**: Retrieves up to `retrieve_k=8` documents based on cosine similarity between query and document embeddings. The top `return_k=3` documents are selected after scoring.

#### Score Normalization
To ensure comparability and interpretability, similarity scores are normalized using the softmax function, which transforms raw scores into probabilities that sum to 1. The formula is:

$$\[
\text{norm\_scores} = \frac{e^{\text{all\_scores}}}{\sum e^{\text{all\_scores}}}
\]$$

Where:
- $\text{all\_scores}$ are $(1.0 - \text{distance})$ the initial similarity scores, calculated as (1.0 - $\text{distance}$), where $(\text{distance})$ is the cosine distance between query and document embeddings (ranging from 0 for identical to 1 for dissimilar).
- The softmax function exponentiates each score ($(e^{\text{all_scores}})$) and divides by the sum of all exponentiated scores ($(\sum e^{\text{all_scores}})$), resulting in normalized scores that represent confidence levels for each document’s relevance.

This normalization ensures that the system prioritizes the most relevant documents, making the retrieval process reliable and transparent.

### Generation
- **Prompt Construction**: Retrieved documents are formatted into a context string, including their content and normalized confidence scores. The prompt instructs the language model to summarize, synthesize, reference standards, and suggest solutions.
- **LLM Generation**: An `OllamaLLM` (model: `qwen2.5:latest`, temperature: 0.3) generates a detailed response based on the context and query, streamed in real-time to the user interface.

### Diagram of the RAG Pipeline
```mermaid
graph LR
    A[User Query] --> B[DPRRetriever]
    B --> C[Query Embedding]
    B --> D[Vector Store (Chroma)]
    C --> E[Similarity Search]
    D --> E
    E --> F[Top-k Documents]
    F --> G[Prompt Construction]
    G --> H[LLM (OllamaLLM)]
    H --> I[Response Generation]
    I --> J[Final Answer]
```

This diagram can be rendered in GitHub to visualize the pipeline from query to response.

## Environment Setup

To run DPR RAG, you’ll need to set up the following:

| Requirement | Description |
|-------------|-------------|
| **Python 3.8 or later** | Ensure Python is installed on your system. Download from [python.org](https://www.python.org/downloads/). |
| **Ollama** | A tool for running large language models locally. Install it based on your operating system: <br> - **Windows**: Download the installer from [Ollama Download](https://ollama.com/download) and run it. <br> - **macOS**: Download the installer from [Ollama Download](https://ollama.com/download), unzip it, and drag the `Ollama.app` to your Applications folder. <br> - **Linux**: Run the installation script as per the [Ollama GitHub repository](https://github.com/ollama/ollama). <br> **Note**: For optimal performance, configure Ollama to use a GPU if available, as the system uses `num_gpu=1` for embeddings. |
| **Python Libraries** | Install the required dependencies listed in `requirements.txt`, including `streamlit`, `langchain`, `chromadb`, `langchain-ollama`, and `numpy`. |

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/armanjscript/DPR-RAG
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd DPR-RAG
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` file includes dependencies like `streamlit`, `langchain`, `chromadb`, `langchain-ollama`, and `numpy`.
4. **Start Ollama**:
   Ensure the Ollama service is running on your system. Follow the instructions from the [Ollama GitHub repository](https://github.com/ollama/ollama) to start the service.
5. **Run the Streamlit App**:
   ```bash
   streamlit run dpr_rag.py
   ```
   This will launch the app in your default web browser.

## Usage

1. **Upload PDFs**:
   - In the sidebar, use the file uploader to select one or more PDF files.
   - Files are saved locally in the `uploaded_pdfs` directory and indexed in a Chroma database (`chroma_db`).
2. **Process Documents**:
   - Click the "Process Documents" button to split, embed, and store the PDFs in the vector store.
3. **Ask Questions**:
   - Enter your query in the chat input field in the main interface.
   - The chatbot retrieves up to 8 relevant document chunks, selects the top 3 based on normalized scores, generates a response, and streams it in real-time.
   - Responses include citations to source documents with confidence scores for reference.
4. **Clear Documents**:
   - Use the "Clear All Documents" button in the sidebar to reset the application by deleting uploaded files and the vector store, with robust error handling for reliability.

## Configuration

The system uses the following fixed parameters for optimal performance:
- **Temperature**: Set to 0.3 for controlled randomness in the language model’s responses, ensuring focused and relevant answers.
- **Retrieval Parameters**: Retrieves up to `retrieve_k=8` documents and returns the top `return_k=3` based on normalized similarity scores.

To customize these parameters, edit the `dpr_rag.py` file to adjust the `OllamaLLM` temperature or the `DPRRetriever` settings (e.g., `retrieve_k` or `return_k`).

## Contributing

We welcome contributions to enhance DPR RAG! To contribute:
- Fork the repository.
- Make your changes in a new branch.
- Submit a pull request with a clear description of your changes.

For detailed guidelines, refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaboration opportunities, please reach out:
- **Email**: [armannew73@gmail.com]
- **GitHub Issues**: Open an issue on this repository for bug reports or feature requests.

## Acknowledgments

This project builds on the following open-source technologies:
- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for document processing and RAG pipeline
- [Chroma](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.com/) for local language models and embeddings
- [NumPy](https://numpy.org/) for numerical operations, including score normalization

## Citations
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://streamlit.io/)
- [Chroma Documentation](https://www.trychroma.com/)
- [Ollama Documentation](https://ollama.com/)

Thank you for exploring DPR RAG! We hope it simplifies your document analysis tasks and inspires further innovation in AI-driven Q&A systems.

#AI #RAG #DPR #PDFQ&A #Streamlit #LangChain #Chroma #Ollama #MachineLearning #NaturalLanguageProcessing