# Import necessary libraries
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read text from a .txt file and return it as a string
def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")  
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store (FAISS) from the text chunks and to save it locally.
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context"\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    # Initialize the ChatGoogleGenerativeAI model with the desired configuration
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3) 
    
    # Define a prompt template for the question-answering task
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the question-answering chain using the specified model, chain type, and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user question and provide an answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the saved vector store for answering questions with dangerous deserialization allowed
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search with the loaded database
    docs = new_db.similarity_search(user_question)

    # Get conversational chain for Q&A
    chain = get_conversational_chain()

    # Get response from the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    # Display the response
    st.write("Reply: ", response["output_text"])


# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat Text File")
    st.header("Chat with Text File Content")
    
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )
    

    user_question = st.text_input("Ask a Question based on the Text File")

    if user_question:
        user_input(user_question)
    #streamlit design
    #adding a slide bar to allow users to upload a text file or multi text files
    with st.sidebar:
        st.title("Menu:")
        txt_docs = st.file_uploader("Upload your .txt Files and Click on the Submit & Process Button", accept_multiple_files=True, type="txt")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Process the text files
                raw_text = get_txt_text(txt_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
