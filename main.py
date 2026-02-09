from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END

# Initialize the Gemini model
# Using gemini-1.5-flash as it is stable and widely supported
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

def chatbot(state: MessagesState):
    """
    Simple chatbot node that invokes the Gemini model.
    """
    return {"messages": [llm.invoke(state["messages"])]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile the graph
graph = builder.compile()

if __name__ == "__main__":
    # Simple test to verify the graph locally
    print("Testing Gemini Graph...")
    from langchain_core.messages import HumanMessage
    result = graph.invoke({"messages": [HumanMessage(content="Hello Gemini!")]})
    print("Response:", result["messages"][-1].content)
