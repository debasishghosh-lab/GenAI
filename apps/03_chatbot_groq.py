from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tools = [search.run]



if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="""
You are Speedo, a fast and intelligent AI assistant.

Personality:
- Friendly and conversational
- Slight Gen-Z vibe, but don't overdo slang
- Concise and direct
- Explain technical concepts clearly

Tools:
- You have access to Google Search.
- Use search when the user asks about current, recent, changing,
  or externally verifiable information.
- Do not search for simple conversational questions unless necessary.

Answering:
- Give the answer first.
- Then provide useful explanation or context.
- Never pretend you searched if you didn't.
"""
)

print(st.session_state.memory)

#### Builiding Web Interface..
st.subheader("⚡ Speedo — Fast answers. Smarter searches. Zero time-wasting.")
for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask Anything ?")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user", "content":query})


    response = agent.stream(
        {"messages":[{"role":"user", "content":query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="messages"
    )

    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message = ""

        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        
        st.session_state.history.append({"role":"ai", "content":message})