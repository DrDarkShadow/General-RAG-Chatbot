import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv(override=True)

# Streamlit app
st.title("General RAG Chatbot")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    st.info("Uploading and processing the PDF file...")
    
    # Save the uploaded file temporarily to disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    st.info("Loading the PDF file...")
    # Create the loader using the temporary file path
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    st.info("Splitting the text into chunks...")
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(docs)

    st.info("Generating embeddings using Hugging Face model...")
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generate embeddings
    text_documents = [chunk.page_content for chunk in text_chunks]
    word_embeddings = embeddings.embed_documents(text_documents)

    st.info("Setting up Chroma for document retrieval...")
    # Set up Chroma with a local directory for persistence
    db = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",  # Local directory to store Chroma data
    )
    
    st.info("Integrating with Azure OpenAI...")
    # Integrate with LLM
    retriever = db.as_retriever()
    llm = AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment="gpt-4o",  # or your deployment
        api_version="2023-09-01-preview",  # or your api version
        temperature=0.6,
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    st.success("PDF processing complete. You can now enter your query.")
    # Query input
    query = st.text_input("Enter your query:")
    if query:
        st.info("Processing your query...")
        response = retrieval_chain.invoke({"input": query})
        answer = response.get("answer", "I don't know the answer.")
        st.write(answer)
else:
    st.write("Please upload a PDF file to proceed.")
