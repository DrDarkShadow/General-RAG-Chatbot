import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
import os
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile
import time
import uuid

# Load environment variables
load_dotenv(override=True)

index_name = "musicbot"
namespace = str(uuid.uuid4())  # Unique namespace for each session

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # CPU optimization for cloud
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_data
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Increased chunk size
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]  # Better PDF processing
    )
    return text_splitter.split_documents(docs)

@st.cache_resource
def load_llm():
    return AzureChatOpenAI(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_deployment="gpt-4o",
        api_version="2023-09-01-preview",
        temperature=0.6,
        streaming=True  # Enable streaming for faster responses
    )

def wait_for_index_ready(pc):
    """Poll index status until ready"""
    start_time = time.time()
    while True:
        try:
            desc = pc.describe_index(index_name)
            if desc.status['ready']:
                return True
            st.info(f"🕒 Index status: {desc.status['state']}")
            time.sleep(5)
        except Exception as e:
            if time.time() - start_time > 300:  # 5 minute timeout
                raise TimeoutError("Index creation timed out")
            time.sleep(5)

st.title("General RAG Chatbot")

# Session state management
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and not st.session_state.processed:
    with st.status("📤 Processing PDF...", expanded=True) as status:
        try:
            # Use temporary directory for better file handling
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "uploaded.pdf")
                
                # Write uploaded file to temporary directory
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process PDF from temp directory
                text_documents = process_pdf(temp_file_path)
                
            st.info(f"📊 Extracted {len(text_documents)} text chunks")

            # Pinecone setup with namespace
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            
            # Check existing index
            if index_name not in pc.list_indexes().names():
                with st.spinner("🏗️ Creating index..."):
                    pc.create_index(
                        name=index_name,
                        dimension=384,
                        metric='cosine',
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    wait_for_index_ready(pc)
            
            # Batch document processing
            embeddings = load_embeddings()
            with st.spinner("🚀 Storing vectors..."):
                PineconeVectorStore.from_documents(
                    documents=text_documents,
                    embedding=embeddings,
                    index_name=index_name,
                    namespace=namespace,  # Use unique namespace
                    batch_size=100,  # Optimized batch size
                    pool_threads=4  # Parallel processing
                )

            # Initialize retriever with namespace
            db = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                namespace=namespace
            )
            retriever = db.as_retriever(
                search_kwargs={'k': 3}  # Optimized result count
            )

            # Create efficient chain
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
                response = st.session_state.retrieval_chain.invoke(
                    {"input": query},
                    config={"max_concurrency": 5}
                )
            
            # Display answer outside the status block
            st.chat_message("assistant").write(response["answer"])
                
            # Show context sources in a separate expander
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
   - Create a vector database
   - Prepare the AI model
3. Ask questions about the document
                    """)