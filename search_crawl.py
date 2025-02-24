from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


tavily_client = TavilyClient(api_key="")
filepath = "queries.txt"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False)

def read_lines(filepath):
  try:
    with open(filepath, 'r', encoding='utf-8') as f:
      return [line.rstrip() for line in f]
  except FileNotFoundError:
    return None



def get_raw_pages():
    queries = read_lines(filepath)
    responses = []
    for query in queries:
        response = tavily_client.search(query, include_raw_content=True)
        responses.extend(response['results'])
    raw_contents = [r['raw_content'] for r in responses]
    return raw_contents


def get_chunks(raw_contents):
    raw_chunks = []
    for i in raw_contents:
        try:
            chunks = text_splitter.split_text(i)
            raw_chunks.extend(chunks)
        except:
            print(i)

