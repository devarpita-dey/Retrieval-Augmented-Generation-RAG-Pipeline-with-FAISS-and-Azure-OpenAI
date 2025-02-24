
# Streamlit UI
import streamlit as st

st.title("RAG-based Search System")
query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        results = vector_store.similarity_search(query=query, k=5)
        search_results = [res.page_content for res in results]
        response = finalize_search_results(query, search_results)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")