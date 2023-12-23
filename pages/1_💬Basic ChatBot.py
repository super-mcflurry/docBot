import streamlit as st
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.llms import Replicate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from streamlit_mic_recorder import speech_to_text
import openai
import time 

@st.cache_resource
def conversation_chain(selected_model):

    memory = ConversationBufferMemory()

    if selected_model == 'ChatGPT-3.5':
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106",temperature=0.5, streaming=True)
    elif selected_model == 'Gemini-Pro':
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7,convert_system_message_to_human=True)
    elif selected_model == 'Llama2-70B':
        llm = Replicate(
        model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 0.5,
            "temperature": 0.5,
            "max_new_tokens": 500,
            "min_new_tokens": -1
            }
        )
    else:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    chain = ConversationChain(llm = llm,memory=memory, verbose=True)

    return chain

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

def text_to_speech(speech_file_path,text):
    response = openai.audio.speech.create(model="tts-1-hd",voice="shimmer",input=text)
    response.stream_to_file(speech_file_path)


def main():
    load_dotenv()

    timestamp = int(time.time())
    speech_file_path = f'audio_response_{timestamp}.mp3'

    st.set_page_config(page_title="DocBot", page_icon=":robot_face:")
    st.header("💬Chat with DocBot")

    with st.sidebar:
        st.subheader("Select your Model")        
        selected_model = st.sidebar.selectbox('Choose a model', ['ChatGPT-3.5', 'Gemini-Pro', 'Llama2-70B', 'FLAN-T5'], key='selected_model')      

        st.subheader("Speech to Text🎙️")    
        voice = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

        st.subheader("Voice Output")        
        voiceSelection = st.sidebar.selectbox('Speech Settings', ['No', 'Yes'], key='voiceSelection')      

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)  

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")

    if voice:
       user_query = voice

    if selected_model:
        chain  = conversation_chain(selected_model)

    if user_query:
        st.chat_message("user").write(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        response = chain.run(user_query)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        if voiceSelection == 'Yes':
            with st.spinner("Processing"):
                text_to_speech(speech_file_path,response)
                st.audio(speech_file_path, format='audio/mp3')
if __name__ == '__main__':
    main()
