from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

# Initialize Gemini LLM and output parser
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
out = StrOutputParser()
chain = llm | out

st.title("⚔️ Query Slayer: Ask Anything ready to slayyyy...😎")
st.markdown("THIS IS A CHATBOT MADE WITH LANGCHAIN SUPPORT..")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask anything")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = chain.invoke(query)
    st.session_state.messages.append({"role": "assistant", "content": res})
    st.chat_message("assistant").markdown(res)