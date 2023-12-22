import os
import utils
import io
import streamlit as st
from streaming import StreamHandler
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.vectorstores import Chroma


from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import OpenAIEmbeddings,  HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from streamlit_mic_recorder import speech_to_text


#Extracts the text from the document
def download_file(file):
    folder = "downloads"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

def get_text(uploaded_files):
    docs = []
    for file in uploaded_files:
        file_path = download_file(file)
        pdf_loader = PyPDFLoader(file_path, extract_images=True)
        docs.extend(pdf_loader.load())
    return docs

#Splits the text into chunks
def split_text(file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function = len,
        is_separator_regex = False
    )
    chunks = text_splitter.split_documents(file)
    return chunks

#Embeds the chunks
def embeddings_vectorstore(chunks):
    # embeddings = OpenAIEmbeddings()
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore
    
#Creates the conversation chain
def conversation_chain(vectorstore):

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    retriever = vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k':2, 'fetch_k':4}
    )
    
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", streaming=True)
    # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,convert_system_message_to_human=True)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

    return chain

def answer_query(query):
    response = st.session_state.conversation({'question': query})
    st.session_state.chat_history = response['chat_history']

    return response['answer']


def main():
    load_dotenv()

    st.set_page_config(page_title="DocBot", page_icon=":robot_face:", layout="wide")
    st.header("Docbot - Chat with your Documents")
       
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
   
     # Display chat input
    user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")

    # Speech-to-Text
    # c1, c2 = st.columns(2)
    # with c1:
       #st.write("Convert speech to text:")
    #with c2:
        #text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

    #if text:
       # user_query = text

    if user_query:
       st.chat_message("user").write(user_query)
       response = answer_query(user_query)
       st.chat_message("assistant").write(response)
        
    with st.sidebar:
        st.subheader("Upload your Documents")
        uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        button = st.button("Process")
        
        if button:
            with st.spinner("Processing"):
                docs = get_text(uploaded_files)
                chunks = split_text(docs)
                vectorstore = embeddings_vectorstore(chunks)
                st.session_state.conversation = conversation_chain(vectorstore)

if __name__ == '__main__':
    main()


