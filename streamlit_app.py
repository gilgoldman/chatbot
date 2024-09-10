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

# Initialize Claude (Anthropic) client with additional header
anthropic_client = Client(
    api_key=claude_api_key,
    default_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
)

# Setup Tavily as a search tool
tavily_search_tool = TavilySearchAPIRetriever(k=3)  # Fetch up to 3 results

def agent_node(state):
    """Node for sending article to Claude for analysis."""
    article = state['article']
    
    # Construct the prompt with the article
    prompt = f"""Article: {article}

Please analyze the article and provide a comprehensive summary. Include key points, main ideas, and any notable insights. If you need to search for additional context or information, please indicate so in your response."""

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=8192,
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
article = st.text_area("Paste your article here:", height=400)

if st.button("Analyze Article"):
    if article:
        st.session_state.article = article
        with st.spinner("Analyzing..."):
            inputs = {
                "messages": [],  # No initial message needed
                "article": st.session_state.article
            }
            result = app.invoke(inputs)
            
            # Display the results
            for message in result['messages']:
                if message.type == "ai":
                    st.write("Analysis:")
                    st.write(message.content)
                elif message.type == "function":
                    st.write("Additional Information:")
                    st.write(message.content)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": message.type, "content": message.content})
    else:
        st.error("Please input an article first.")

# Display analysis history
st.subheader("Analysis History")
for message in st.session_state.chat_history:
    st.write(f"{message['role'].capitalize()}:")
    st.write(message.get('content', ''))
    st.write("---")

# Clear history button
if st.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.article = ""
    st.success("History cleared and article removed.")