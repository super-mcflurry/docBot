import os
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma

from langchain.llms import Replicate
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

from streamlit_mic_recorder import speech_to_text
import openai
import time

docs = []

# Extracts the text from the document
def download_file(file):
    folder = "downloads"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f"./{folder}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path


def get_text(uploaded_files):
    for file in uploaded_files:
        file_path = download_file(file)
        if file.name.endswith(".pdf"):
            pdfReader(file_path)
        elif file.name.endswith(".docx"):
            docxReader(file_path)
        elif file.name.endswith(".txt"):
            txtReader(file_path)


def pdfReader(file_path):
    pdf_loader = PyPDFLoader(file_path, extract_images=True)
    docs.extend(pdf_loader.load())


def docxReader(file_path):
    doc = Docx2txtLoader(file_path)
    docs.extend(doc.load())


def txtReader(file_path):
    txt = TextLoader(file_path)
    docs.extend(txt.load())


# Splits the text into chunks
def split_text(file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(file)
    return chunks


# Embeds the chunks
def embeddings_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore


# Creates the conversation chain
def conversation_chain(vectorstore, selected_model):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    if selected_model == "GPT-3.5-Turbo":
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-1106", temperature=0.5, streaming=True
        )
    elif selected_model == "GPT-4":
         llm = ChatOpenAI(
            model_name="gpt-4-1106-preview", temperature=0.5, streaming=True
        )
    elif selected_model == "Gemini-Pro":
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", temperature=0.5, convert_system_message_to_human=True
        )
    elif selected_model == "Llama2-70B":
        llm = Replicate(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={
                "debug": False,
                "top_k": 50,
                "top_p": 0.5,
                "temperature": 0.5,
                "max_new_tokens": 500,
                "min_new_tokens": -1,
            },
        )
    else:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )

    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    return chain


def answer_query(query):
    start_time = time.time()
    response = st.session_state.conversation({"question": query})
    end_time = time.time()
    response_time = end_time - start_time
    print(f"Response Time: {response_time:.2f} seconds\n")
    return response["answer"]


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]


def check_internet_connection():
    try:
        # Try to make an HTTP request to a reliable server (e.g., Google)
        response = requests.get("http://www.google.com", timeout=5)
        response.raise_for_status()  # Raise an error for bad responses
        return True
    except requests.RequestException:
        pass
    return False


def text_to_speech(speech_file_path, text):
    response = openai.audio.speech.create(model="tts-1-hd", voice="shimmer", input=text)
    response.stream_to_file(speech_file_path)


def main():
    load_dotenv()

    timestamp = int(time.time())
    speech_file_path = f"audio_response_{timestamp}.mp3"

    st.set_page_config(page_title="DocBot", page_icon=":robot_face:")
    st.header("📝Chat with Your Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    with st.sidebar:
        st.subheader("Upload your Documents")
        uploaded_files = st.file_uploader(
            label="Upload files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

        selected_model = st.sidebar.selectbox(
            "Choose a model",
            ["GPT3.5-Turbo","GPT-4" "Gemini-Pro", "Llama2-70B", "FLAN-T5"],
            key="selected_model",
        )
        button = st.button("Process")

        if button:
            with st.spinner("Processing"):
                get_text(uploaded_files)
                chunks = split_text(docs)
                vectorstore = embeddings_vectorstore(chunks)
                st.session_state.conversation = conversation_chain(
                    vectorstore, selected_model
                )
                st.success("Done!")

        st.subheader("Speech to Text🎙️")
        voice = speech_to_text(
            language="en", use_container_width=True, just_once=True, key="STT"
        )

        st.subheader("Voice Output")
        voiceSelection = st.sidebar.selectbox(
            "Speech Settings", ["No", "Yes"], key="voiceSelection"
        )

        st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")

    if voice:
        user_query = voice

    if user_query:
        if check_internet_connection():
            st.chat_message("user").write(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})
            response = answer_query(user_query)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
            if voiceSelection == "Yes":
                with st.spinner("Processing"):
                    text_to_speech(speech_file_path, response)
                    st.audio(speech_file_path, format="audio/mp3")

        elif check_internet_connection == False:
            st.error("Please check your internet connection!", icon="🚨")


if __name__ == "__main__":
    main()
