from typing import List, Any

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def read_data_from_pdf(pdf_docs: List[str]) -> str:
    """Reads text from PDF files and concatenates them."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def read_data_from_txt(txt_files: List[str]) -> str:
    """Reads text from text files and concatenates them."""
    text = ""
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            text += file.read()
    return text



def get_text_chunks(text: str) -> List[str]:
    """The purpose of this function is to split a long text into smaller, more 
    manageable chunks. This can be useful when dealing with large texts, 
    such as documents or articles, especially in scenarios where processing or 
    analyzing large texts is required."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks: List[str]) -> None:
    """The purpose of this function is to generate vector representations for 
    each text chunk provided and then store these vectors using FAISS for 
    efficient similarity search and retrieval."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


def get_conversation_chain() -> Any:
    """
    This function initializes a conversation chain for answering questions based on a provided context.
    
    Prompt Template:
    ----------------
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Don't provide the wrong answer.
    
    Context:
    {context}?
    
    Question: 
    {question}
    
    Answer:
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Don't provide the wrong answer.

    Context:
    {context}?

    Question: 
    {question}

    Answer
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperatue=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def get_response_to_user_question(user_question: str) -> None:
    """Gets a response to a user question."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    

def get_pdf_raw_text() -> str:
    """Gets raw text from a PDF file."""
    pdf_files: List[str] = [os.getcwd() + "/information.pdf"]  # Provide your PDF file paths here
    raw_text = ""
    for pdf_file in pdf_files:
        with open(pdf_file, "rb") as f:
            pdf_docs: List[bytes] = [f]
            raw_text += read_data_from_pdf(pdf_docs)
    return raw_text

def get_text_file_raw_text() -> str:
    txt_files: List[str] = [os.getcwd() + "/information.txt"]  # Provide your .txt file paths here
    raw_text = ""
    for txt_file in txt_files:
        raw_text += read_data_from_txt([txt_file])
    return raw_text

def main() -> None:
    message: str = "Invalid file type provided. Choode either (txt or pdf)."
    choose_file_to_read_from : str = input("Please select file type (txt or pdf) to search information from: ")
    if not choose_file_to_read_from: raise ValueError(message)
    if choose_file_to_read_from.lower() == 'txt':
        raw_text = get_text_file_raw_text()
    elif choose_file_to_read_from == 'pdf':
        raw_text = get_pdf_raw_text()
    else:
        raise ValueError(message)
    
    text_chunks: List[str] = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    while True:
        user_question: str = input("Ask a Question (or type 'exit' to quit): ")
        if user_question == "exit":
            break
        get_response_to_user_question(user_question)


if __name__ == "__main__":
    main()
