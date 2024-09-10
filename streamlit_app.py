import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage
from anthropic import Client

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'article' not in st.session_state:
    st.session_state.article = ""

# Set API keys securely using Streamlit secrets
tavily_api_key = st.secrets["TAVILY_KEY"]
claude_api_key = st.secrets["CLAUDE_KEY"]

# Ensure API keys are provided
if not tavily_api_key or not claude_api_key:
    st.error("API keys for Tavily and Claude must be set in Streamlit secrets.")
    st.stop()

# Initialize Claude (Anthropic) client
anthropic_client = Client(api_key=claude_api_key)

# Setup Tavily as a search tool
tavily_search_tool = TavilySearchAPIRetriever(k=3)  # Fetch up to 3 results

def agent_node(state):
    """Node for sending query to Claude using the messages API."""
    messages = state['messages']
    article = state['article']
    
    # Construct the prompt with the article and the user's query
    prompt = f"""Article: {article}

User Query: {messages[-1].content}

Please analyze the article and answer the user's query. If you need to search for additional information, please indicate so in your response."""

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    if response and response.content and hasattr(response.content[0], 'text'):
        text_content = response.content[0].text
    else:
        text_content = "No valid response received."
    
    return {"messages": [AIMessage(content=text_content)]}

def action_node(state):
    """Action node to invoke Tavily Search."""
    messages = state['messages']
    query = messages[-1].content
    result = tavily_search_tool.invoke(query)
    return {"messages": [FunctionMessage(content=str(result), name="TavilySearch")]}

def should_continue(state):
    """Determine if we need to continue based on Claude's response."""
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and "search" in last_message.content.lower():
        return "continue"
    return "end"

class State(TypedDict):
    messages: List[HumanMessage | AIMessage | FunctionMessage]
    article: str

# Create the LangGraph workflow
state_graph = StateGraph(State)
state_graph.add_node("agent", agent_node)
state_graph.add_node("action", action_node)
state_graph.set_entry_point("agent")

state_graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END}
)
state_graph.add_edge("action", "agent")

app = state_graph.compile()

# Streamlit UI
st.title("Article Analysis with LangGraph")

# Article input
st.subheader("Input Article")
article = st.text_area("Paste your article here:", height=300)
if st.button("Save Article"):
    st.session_state.article = article
    st.success("Article saved successfully!")

# User query input
st.subheader("Ask a Question")
user_input = st.text_input("Enter your query about the article:")

if st.button("Analyze"):
    if st.session_state.article and user_input:
        with st.spinner("Analyzing..."):
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "article": st.session_state.article
            }
            result = app.invoke(inputs)
            
            # Display the results
            for message in result['messages']:
                if message.type == "human":
                    st.write(f"You: {message.content}")
                elif message.type == "ai":
                    st.write(f"AI: {message.content}")
                elif message.type == "function":
                    st.write(f"Search Results: {message.content}")
                
                # Add to chat history
                st.session_state.chat_history.append({"role": message.type, "content": message.content})
    elif not st.session_state.article:
        st.error("Please input an article first.")
    else:
        st.error("Please enter a query.")

# Display chat history
st.subheader("Analysis History")
for message in st.session_state.chat_history:
    st.write(f"{message['role'].capitalize()}: {message.get('content', '')}")

# Clear history button
if st.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.article = ""
    st.success("History cleared and article removed.")