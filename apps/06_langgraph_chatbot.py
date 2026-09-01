from email import message
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import START,END,StateGraph
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

class Scheema(BaseModel):
    messages:Annotated[list,add_messages]

llm=ChatGroq(model="openai/gpt-oss-20b")

def LLMNODE(State:Scheema):
    res=llm.invoke(State.messages)
    State.messages=[res]
    return State

Memory=InMemorySaver()

graph=StateGraph(Scheema)

graph.add_node("LLM",LLMNODE)
graph.add_edge(START,"LLM")
graph.add_edge("LLM",END)

final_graph=graph.compile(checkpointer=Memory)

while True:
    query=input("user:")
    res=final_graph.invoke({"messages":[{"role":"user","content":query}]},
                    {"configurable":{"thread_id":"123@"}}
    )
    print(res["messages"][-1].content)
    if query.lower() in ['quit','stop','bye']:
        break