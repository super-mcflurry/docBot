import streamlit as st
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from streamlit_mic_recorder import speech_to_text


def answer_query(query):
    response = st.session_state.conversation({'question': query})
    return response['answer']

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]


def main():
    load_dotenv()

    st.set_page_config(page_title="DocBot", page_icon=":robot_face:")
    st.header("📊Chat with Your Data")
       
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
 
    # to show chat history on ui
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])
   

    with st.sidebar:
        st.subheader("Upload your Documents")
        csv_files = st.file_uploader(label='Upload CSV files', type=['csv'], accept_multiple_files=True)
        
        if csv_files:
                agent = create_csv_agent(ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                                         csv_files,
                                         verbose=True,
                                         agent_type=AgentType.OPENAI_FUNCTIONS,
                                         handle_parsing_errors=True
                                        )

        st.subheader("Speech to Text🎙️")    
        voice = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history) 
        
    user_query = st.chat_input(placeholder="Ask me anything!", key="user_input")
    voice = speech_to_text(language='en', just_once=True, key='STT')

    if voice:
       user_query = voice
       
    if user_query:
       st.chat_message("user").write(user_query)
       st.session_state.messages.append({"role": "user", "content": user_query})
       response = agent.run(user_query)
       st.session_state.messages.append({"role": "assistant", "content": response})
       st.chat_message("assistant").write(response)
        

if __name__ == '__main__':
    main()


