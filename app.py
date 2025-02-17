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

# Session state management
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'chroma_dir' not in st.session_state:
    st.session_state.chroma_dir = None

uploaded_file = st.file_uploader("Choose any Pdf file", type="pdf")

if uploaded_file is not None and not st.session_state.processed:
    with st.status("📤 Processing PDF...", expanded=True) as status:
        try:
            # Cleanup previous session
            if st.session_state.chroma_dir:
                try:
                    shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)
                except Exception as e:
                    pass

            # Create shorter temp directory path
            temp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix="st_")
            st.session_state.chroma_dir = temp_dir

            # Process PDF
            temp_file_path = os.path.join(temp_dir, "doc.pdf")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            text_documents = process_pdf(temp_file_path)

            # Store vectors
            embeddings = load_embeddings()
            db = Chroma.from_documents(
                documents=text_documents,
                embedding=embeddings,
                persist_directory=temp_dir
            )

            # Initialize retriever
            db = Chroma(
                persist_directory=temp_dir,
                embedding_function=embeddings
            )
            retriever = db.as_retriever(search_kwargs={'k': 3})

            # Create chain
            llm = load_llm()
            prompt = ChatPromptTemplate.from_template("""
            Answer using only this context:
            {context}
            Question: {input}
            """)
            
            st.session_state.retrieval_chain = create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(llm, prompt)
            )

            st.session_state.processed = True
            status.update(label="✅ Ready for queries!", state="complete")

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")
            st.session_state.processed = False

if st.session_state.processed:
    query = st.chat_input("Ask about the document:")
    if query:
        try:
            with st.status("🔍 Processing...", expanded=False):
                response = st.session_state.retrieval_chain.invoke({"input": query})
            
            st.chat_message("assistant").write(response["answer"])
            
            with st.expander("📚 See sources"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', '?')})")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()

        except Exception as e:
            st.error(f"❌ Query failed: {str(e)}")

# Cleanup when session ends
if not st.session_state.processed and st.session_state.chroma_dir:
    try:
        shutil.rmtree(st.session_state.chroma_dir, ignore_errors=True)
    except:
        pass

st.sidebar.markdown("""
**How it works:**
1. Upload a PDF document
2. The system will:
   - Extract text content
   - Create local vector database
   - Prepare the AI model
3. All data deleted after session ends
""")
