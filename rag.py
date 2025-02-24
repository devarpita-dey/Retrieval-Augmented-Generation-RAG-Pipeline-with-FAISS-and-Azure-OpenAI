import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from embedding import EmbeddingModel
import json
from search_crawl import get_raw_pages, get_chunks
from uuid import uuid4
from langchain_core.documents import Document
import os
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2023-06-01-preview",  # or your api version
    temperature=0)
model = EmbeddingModel()

index = faiss.IndexFlatL2(len(model.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

get_content = get_raw_pages()
raw_contents = [r['raw_content'] for r in get_content]
raw_chunks = get_chunks(raw_contents)

def finalize_search_results(query:str,search_results)->str:
    results = json.dumps(search_results)
    messages = [
        ("system",
            "You are a helpful assistant that summarizes search results and given query, list of search results. \
            Provide response from the search results only. If required provide the response in markdown",
        ),
        ("human",
         f"user_query: {query} \
         search_results: {results}")
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content

documents=[]
for i in raw_chunks:
  documents=documents+[Document(page_content=i)]

uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)




query = "What is  the net worth of Neymar?"
results = vector_store.similarity_search(query=query,k=5)
search_results = []
for res in results:
    search_results.append(res.page_content)

finalize_search_results("What is  the net worth of Neymar?", search_results)



