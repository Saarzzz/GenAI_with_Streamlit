import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document  
import pandas as pd

# Load environment variables
load_dotenv()

# Load the GROQ and Gemini API keys
groq_api_key = os.getenv('GROQ_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

st.title("Tembusu Grand QnA System")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", temperature=0.4)

prompt = ChatPromptTemplate.from_template(
"""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}
    Always follow these guidelines:
    1. Provide a brief description without using phrases like "Based on the provided context."
    2. Be as detailed as possible.
    3. Always use the <br> tag for line breaks instead of '\n' in the output.
    4. When providing a link, use the <a href='link' target='_blank'> tag for redirection.
    5. Do not use information other than the provided context.
    """
)

def read_pdf(file_path): #Reading the PDF Files
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def read_docx(file_path): #Reading the DOCX Files
    doc = DocxDocument(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

def read_image(file_path): #Reading the Image Files
    image = Image.open(file_path)
    return file_path, os.path.basename(file_path) 

def read_xlsx(file_path): #Reading the XLSX Files
    df = pd.read_excel(file_path)
    text = df.to_string(index=False)
    return text

#Loading the Files 
def load_documents(folder_path):
    documents = []
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            text = read_pdf(file_path)
            documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith('.docx'):
            text = read_docx(file_path)
            documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith('.xlsx'):
            text = read_xlsx(file_path)
            documents.append(Document(page_content=text, metadata={"source": filename}))
        elif filename.endswith('.png'):
            image_path, image_name = read_image(file_path)
            documents.append(Document(page_content=image_path, metadata={"source": image_name}))
            print(filename)
    return documents

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        folder_path = 'C:/Users/Admin/Downloads/Tembusu_Grand/'
        st.session_state.docs = load_documents(folder_path)  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

if st.button("Load the Embeddings"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

prompt1 = st.text_input("Enter Your Question:-")

import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time:", time.process_time() - start)
    st.write(response['answer'])
    displayed_image = False
    for doc in response["context"]:
        if doc.metadata["source"].endswith('.png') and not displayed_image:
            st.image(doc.page_content, caption=doc.metadata["source"])
            displayed_image = True
         
    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

      # Display images based on file names in the response
        for doc in response["context"]:
          if doc.metadata["source"].endswith('.png'):
            st.image(doc.page_content, caption=doc.metadata["source"])
