from streamlit import chat_input
from IPython.core.debugger import prompt
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

out=StrOutputParser()
chain=llm | out



# while True:
#     query=input("user:")

#     if query.lower() in ["quit","exit","bye"]:
#         print("Goodbye hope I answered your query correctly!!!!!!!!")
#         break

#     res=chain.invoke(query)
#     print("AI:",res)

st.title("⚔️Query Slayer:Ask Anything ready to slayyyy...😎")
st.markdown("THIS IS A CHATBOT MADE WITH LANGCHIN SUPPORT..")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)

query=st.chat_input("Ask anything")
if(query):
    st.session_state.messages.append({"role":"user","content":query})
    st.chat_message("user").markdown(query)
    res=chain.invoke(query)
    st.session_state.messages.append({"role":"AI","content":res})
    st.chat_message("AI").markdown(res)