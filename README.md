# General RAG Chatbot
The General RAG Chatbot is a Retrieval-Augmented Generation (RAG)-based chatbot that allows users to upload PDF documents and ask queries about their content. It utilizes LangChain for document processing, Hugging Face embeddings for text representation, ChromaDB for storage and retrieval, and Azure OpenAI GPT-4 for answering user queries.

### Features  
 ✅ Upload PDFs – Users can upload a PDF to extract text for querying.   
 ✅ Text Chunking – The document is split into smaller, meaningful text chunks.  
 ✅ Embeddings Generation – Converts text into vector embeddings using Hugging Face’s all-MiniLM-L6-v2 model.  
 ✅ Efficient Retrieval – Uses ChromaDB for storing and retrieving relevant text sections.  
 ✅ AI-Powered Responses – Retrieves the best-matching content and provides answers using Azure OpenAI GPT-4.  
 ✅ User-Friendly Interface – Built using Streamlit for easy interaction.  

# How It Works
1. Upload a PDF: The system reads and processes the document.
2. Generate Embeddings: Text is split into chunks and transformed into embeddings.
3. Ask a Question: Enter a query based on the document’s content.
4. Receive AI-Powered Answers: The chatbot retrieves the most relevant information and generates an accurate response.

# Screenshots
1. Uploading a PDF File
   ![1](https://github.com/user-attachments/assets/6c54db92-af09-4525-b110-37ef1755e11f)

2. Chatbot Query and Response
   ![2](https://github.com/user-attachments/assets/ca067638-b29f-4387-906f-2be5c40013f9)


# Installation and Setup
Prerequisites
Ensure you have Python installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

# Environment Setup
Create a .env file and add your Azure OpenAI API Key:
```bash
AZURE_OPENAI_API_KEY=your_api_key_here
```
# Running the Chatbot
Run the Streamlit app using:

```bash
streamlit run app.py
```

# Technologies Used
- Python – Backend logic
- Streamlit – User interface
- LangChain – Document processing and retrieval
- ChromaDB – Storing and retrieving document embeddings
- Hugging Face – Generating vector embeddings
- Azure OpenAI GPT-4 – Answer generation
  
# Future Improvements
- Support for multiple document formats (DOCX, TXT)
- Enhanced retrieval for large documents
- Additional model options for improved response quality
  
# Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

# License
This project is licensed under the MIT License.
