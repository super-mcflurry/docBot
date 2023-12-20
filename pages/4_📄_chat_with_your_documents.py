import os
import utils
import streamlit as st
from streaming import StreamHandler
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch


from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import OpenAIEmbeddings,  HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from streamlit_mic_recorder import speech_to_text

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your documents')

class Chatbot:

    def download_file(self,file):
        folder = "downloads"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path
    
    def get_text(self, uploaded_files):
        docs = []
        for file in uploaded_files:
            file_path = self.download_file(file)
            pdf_loader = PyPDFLoader(file_path)
            docs.extend(pdf_loader.load())
        return docs

    def split_text(self, file):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(file)
        return chunks
    
    def embeddings_vectorstore(self, chunks):
       embeddings = OpenAIEmbeddings()
       # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
       # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

       vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)
       return vectorstore
       
    def conversation_chain(self,vectorstore):

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", streaming=True)
        # llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,convert_system_message_to_human=True)
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
 
        chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

        return chain
    
    @utils.enable_chat_history
    def main(self):
        load_dotenv()
        with st.sidebar:
            st.subheader("Upload your Documents")
            uploaded_files = st.file_uploader(label='Upload PDF files', type=['pdf'], accept_multiple_files=True)
        if not uploaded_files:
            st.error("Please upload PDF documents to continue!")
            st.stop()

        # Chatbot
        if uploaded_files:
            with st.spinner("Processing"):
                docs = self.get_text(uploaded_files)
                chunks = self.split_text(docs)
                vectorstore = self.embeddings_vectorstore(chunks)
                qa_chain = self.conversation_chain(vectorstore)

        # Display chat input
        user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")

        # Speech-to-Text
        c1, c2 = st.columns(2)
        with c1:
            st.write("Convert speech to text:")
        with c2:
            text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

        # Update chat input with the latest speech-to-text result
        st.session_state.text_received = text

        if text:
            user_query = text

        if user_query:
            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Chatbot()
    obj.main()
