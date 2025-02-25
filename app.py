import streamlit as st
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv(override=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)

@st.cache_resource
def load_llm():
    return AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment="gpt-4o",
        api_version="2023-09-01-preview",
        temperature=0.6,
        streaming=True
    )

st.markdown("<h1 style='text-align: center;'>Local RAG Chatbot</h1>", unsafe_allow_html=True)

def get_file_hash(content):
    return hashlib.sha256(content).hexdigest()

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Generate unique ID based on file content
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)

    # Create persistent storage path
    base_dir = os.path.join(tempfile.gettempdir(), "rag_chatbot")
    if "chroma_dir" not in st.session_state:
        st.session_state.chroma_dir = os.path.join(base_dir, file_hash, "chroma")
    pdf_path = os.path.join(base_dir, file_hash, "doc.pdf")

    # Create directory structure
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Save PDF if not exists
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(file_content)

    # Process PDF and create vector store if not exists
    if not os.path.exists(st.session_state.chroma_dir):
        with st.status("📤 Processing PDF...", expanded=True):
            docs = process_pdf(pdf_path)
            embeddings = load_embeddings()
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=st.session_state.chroma_dir
            )

    # Load existing vector store
    embeddings = load_embeddings()
    db = Chroma(
        persist_directory=st.session_state.chroma_dir,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={'k': 3})
    
    # Create conversation chain
    llm = load_llm()
    prompt = ChatPromptTemplate.from_template("""
    Answer using only this context:
    {context}
    Question: {input}
    """)
    
    retrieval_chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt)
    )
    
    # Query interface
    query = st.chat_input("Ask about the document:")
    if query:
        try:
            with st.status("🔍 Processing...", expanded=False):
                response = retrieval_chain.invoke({"input": query})
            
            st.chat_message("assistant").write(response["answer"])
            
            with st.expander("📚 See sources"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', '?')})")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()
        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")

st.sidebar.markdown("""
**How it works:**
1. Upload a PDF document
2. The system will:
   - Extract text content
   - Create local vector database
   - Prepare the AI model
3. All data persists in temp directory until system cleanup
""")
