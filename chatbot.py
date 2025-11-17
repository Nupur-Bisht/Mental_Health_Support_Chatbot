import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage ,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#injection-load ,split and store
def injection():
  loader=PyPDFLoader('mental_health.pdf')
  document =loader.load()

  text_spitter=RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
  document_chunks=text_spitter.split_documents(document)
  document_chunks = document_chunks[:150]

  vector_store = Chroma.from_documents(
    documents=document_chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"  
)

  return vector_store

#conversation with AI 
def get_conversation(retriever_chain):
    llm =ChatOpenAI()
    prompt=ChatPromptTemplate.from_messages([
        ("system","Answer the question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),

    ])
    stuff =create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain,stuff)
     


# retrieve information

def get_retriever(vector_store):
     llm = ChatOpenAI(api_key=api_key)
     retriever= vector_store.as_retriever()
     prompt=ChatPromptTemplate.from_messages([
         MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation , generate a search query to look up in order to get information relevant to conversation")

     ])
     retriever_chain= create_history_aware_retriever(llm,retriever,prompt)
     return retriever_chain

   
st.set_page_config(page_title='Mental Health Website') 
if"chat_history" not in st.session_state:
     st.session_state.chat_history=[]

st.title("Mental Health Care ")


vector_store = injection()
retriever_chain=get_retriever(vector_store)
conversation= get_conversation(retriever_chain)

user_query=st.chat_input("Ask Question Related to Mental Health...")
if user_query is not None and user_query !="":
          ai_response=conversation.invoke({
         "chat_history":st.session_state.chat_history,
         "input":user_query
          })
         # st.markdown(ai_response)
          st.session_state.chat_history.append(HumanMessage(content=user_query))
          st.session_state.chat_history.append(AIMessage(content=ai_response["answer"]))

for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)






