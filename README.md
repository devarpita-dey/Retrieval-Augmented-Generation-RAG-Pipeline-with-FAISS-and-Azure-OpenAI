# Retrieval-Augmented-Generation-RAG-Pipeline-with-FAISS-and-Azure-OpenAI


This repository contains a pipeline for web scraping, document indexing, and semantic search using FAISS and Azure OpenAI. It integrates web crawling with Tavily, text embedding using Sentence Transformers, and retrieval-based summarization using GPT (gpt-4o).

**Features**

Web scraping using **Tavily API**
Chunking text using LangChain's **RecursiveCharacterTextSplitter**
Embedding using **sentence-transformers**
Indexing and similarity search using **FAISS**
Retrieval-augmented generation (RAG) using **Azure OpenAI**

**Installation**

1. Clone the repository:

git clone [https://github.com/yourusername/repo-name.git
cd repo-name](https://github.com/devarpita-dey/Retrieval-Augmented-Generation-RAG-Pipeline-with-FAISS-and-Azure-OpenAI)

2. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Configuration

Set up environment variables:

export TAVILY_API_KEY="your-tavily-api-key"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"

5.Set up input queries:
Create a queries.txt file containing search queries (one per line).

6.Running the Scraper
To fetch raw web pages:

python search_crawl.py

This script:
a> Reads queries from queries.txt
b> Fetches search results using Tavily API
c>Extracts raw content from search results

**d Indexing Documents**

To generate embeddings and create a FAISS index:
python index_documents.py

This script:
Loads raw content
Splits text into smaller chunks
Generates embeddings using SentenceTransformer
Stores vectors in FAISS index
Running a Search Query

**To perform similarity search and retrieve results:**
python search_rag.py --query "What is the net worth of Neymar?"

This script:
Embeds the query
Searches the FAISS index
Retrieves relevant documents
Uses GPT-4 to summarize search results
