import streamlit as st
from langchain_astradb.graph_vectorstores import AstraDBGraphVectorStore
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Streamlit Page Setup
st.set_page_config(page_title="2024 Budget RAG App", layout="wide")
st.title("2024 Interim Union Budget RAG App")

# Sidebar for User Input
st.sidebar.header("User Input")
query = st.sidebar.text_input("Enter your query:")

# API Tokens
token = st.secrets("ASTRA_DB_TOKEN")
hf_api_token = st.secrets("ASTRA_DB_ID")

# HuggingFace Model Setup
model_name = "sentence-transformers/all-mpnet-base-v2"
hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# AstraDB Setup
store = AstraDBGraphVectorStore(
    embedding=hf_embedding,
    token=token,
    api_endpoint="https://12a9c05c-79a9-48e8-acaa-53cac4c16854-us-east-2.apps.astra.datastax.com",
    collection_name="interim_budget"
)

# LLM Setup
llm = HuggingFaceEndpoint(
    repo_id="Qwen/QwQ-32B-Preview",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.02,
    huggingfacehub_api_token=hf_api_token
)

# Prompt Template
template = """Based on the retrieved context and the user's query, provide a concise and highyly in-depth answer regarding the Indian Interim Union Budget.
### Query:
{query}

### Context:
{context}

Helpful Answer:

"""


custom_rag_prompt = PromptTemplate.from_template(template)

if query:
    # Retrieve relevant documents
    retriever = store.as_retriever(search_type="mmr_traversal")
    docs = retriever.get_relevant_documents(query)

    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    logging.info(f"Context: {context}")

    # Format the prompt
    formatted_prompt = custom_rag_prompt.format(context=context, query=query)

    # Generate LLM Response
    response = llm.invoke(formatted_prompt)

    # Display the Result
    st.subheader("Structured Explanation:")
    st.markdown(response)
else:
    st.info("Please enter a query in the sidebar.")
