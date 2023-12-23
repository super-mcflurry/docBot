import streamlit as st

st.set_page_config(
    page_title="DocBot",
    page_icon='robot_face:',
    layout='wide'
)

st.header("Chat with DocBot")

st.write("""

DocBot: Your Intelligent Document Assistant

DocBot is an advanced chatbot designed to streamline your document-related tasks with unparalleled efficiency. With its versatile features, DocBot can seamlessly engage in conversations, analyze basic language models, and effortlessly handle various document formats, including CSV files, PDFs, TXT, and DOCX.

- **Basic Chatbot**: DocBot engages users in natural and dynamic conversations, utilizing sophisticated language models to provide intuitive and responsive interactions.

- **Chat with CSV:**: DocBot simplifies data-driven decision-making by effortlessly navigating and analyzing CSV files, allowing users to extract valuable insights and perform calculations with ease.

- **Chat with Documents (PDF, TXT, DOCX)**: DocBot revolutionizes document management, extracting and organizing information from PDF, TXT, and DOCX files, providing users with a streamlined workflow for efficient analysis and reference.
""")
