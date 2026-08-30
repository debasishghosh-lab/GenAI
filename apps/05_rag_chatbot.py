from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import streamlit as st

@st.cache_resource
def create_vector_store():
    loader = PyPDFLoader(
        r"D:\GEN_AI\data\data_science_syllabus.pdf"
    )

    document = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    pdf_chunks = splitter.split_documents(document)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    return Chroma.from_documents(
        documents=pdf_chunks,
        embedding=embeddings
    )


vector_store = create_vector_store()

llm=ChatGroq(model="openai/gpt-oss-20b")


def get_context(query:str):
    data=vector_store.similarity_search(query)
    context=""
    for d in data:
        context=context+d.page_content+"\n"
    
    return{
        "context":context,
        "question":query
    }

prompt = PromptTemplate.from_template(
'''
You are Connie, a friendly and supportive AI assistant for this document.

You have two modes of interaction:

 1. CASUAL CONVERSATION
For greetings, goodbyes, thanks, small talk, or questions about yourself
(e.g., "hi", "hello", "how are you?", "who are you?", "what can you do?"),
respond naturally and briefly.

Casual conversation does NOT require information from the context.

 2. DOCUMENT-BASED QUESTIONS
For questions asking about information, concepts, facts, details, numbers,
or topics related to the document:

- Use ONLY the information provided in the context.
- Do NOT use outside knowledge.
- Do NOT guess, assume, or invent information.
- If the answer is not supported by the context, respond exactly:
  "No Context Provided"
- Answer the user's question directly.
- Do not provide unnecessary information.
- Follow the user's requested format or length, such as "in one line",
  "briefly", "explain in detail", or "give 3 points".
- Preserve names, dates, prices, percentages, durations, and numerical
  values exactly as provided in the context.
- If the answer is spread across multiple context sections, combine the
  relevant information.
- If the context contains conflicting information, clearly mention the
  conflict instead of guessing.

 IMPORTANT
Being friendly does NOT mean answering document-related questions using
your general knowledge.

For document-related questions, the context is your ONLY source of truth.

Keep your responses clear, natural, and conversational.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
'''
)

def is_casual(query):

    casual_words = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "thanks",
        "thank you",
        "bye",
        "who are you",
        "what can you do"
    ]

    query = query.lower().strip()

    return any(word in query for word in casual_words)



st.subheader("Connie Here🤺-Get answers from your context")

rag_chain=get_context|prompt|llm

if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


query=st.chat_input("Know your doc..")
if query:

    st.chat_message("user").markdown(query)

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    if is_casual(query):

        response = llm.invoke(
            f"""
            You are Connie, a friendly document assistant.

            Respond naturally and briefly to this casual message:

            {query}
            """
        )

    else:

        response = rag_chain.invoke(query)

    st.chat_message("assistant").markdown(response.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response.content
    })