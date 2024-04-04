import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import faiss
from langchain_openai import OpenAI,ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_openai import OpenAIEmbeddings
import requests
import os

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        text = response.text
        return text
    else:
        return None


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory, chain_type="stuff"
    )
    return conversation_chain


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with UR Decumntations", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:

        st.header("Choose UR Docs :books:")

        file_type = st.selectbox("Select file type", ["PDF", "URL"])

        if file_type == "PDF":
            pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
            if st.button(
                "Process PDF file",
            ):
                if pdf_file is not None:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_file)
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(
                            vectorstore
                        )

        elif file_type == "URL":
            url = st.text_input("Enter URL")
            if st.button("Process URL"):
                with st.spinner("Processing..."):
                    if url:
                        raw_text = get_text_from_url(url)
                        if raw_text is not None:
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore
                            )
                        else:
                            st.error(
                                "Failed to fetch text from URL. Please make sure the URL is correct."
                            )

    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_user_input(user_question)


if __name__ == "__main__":
    main()

    ################################################################
