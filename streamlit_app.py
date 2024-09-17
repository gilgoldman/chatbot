import streamlit as st
from langchain_community.tools import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain_core.messages import AIMessage

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Set API keys securely using Streamlit secrets
tavily_api_key = st.secrets["TAVILY_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]

# Ensure API keys are provided
if not tavily_api_key or not claude_api_key:
    st.error("API keys for Tavily and Claude must be set in Streamlit secrets.")
    st.stop()

# Tavily tool setup
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

# Anthropic model setup with headers for the beta feature
llm = ChatAnthropic(
    api_key=claude_api_key,  # API key passed here
    model="claude-3-5-sonnet-20240620",
    default_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
)

# Define the static part of the system prompt
SYSTEM_PROMPT = '''
    You are an expert and diligent content editor with years of experience at leading news outlets.
    You inspect every article that you review carefully and diligently, one paragraph at a time, and then as a whole.
    You take your time to think through an article step by step before suggesting a correction. 
    You have infinite patience and considerable attention to detail. No mistake gets past you.
'''

# Prompt setup with placeholders
prompt = ChatPromptTemplate(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{user_input}"),  # This will be dynamically filled
        ("placeholder", "{messages}"),
    ]
)

# Tool chaining setup
llm_with_tools = llm.bind_tools([tool])
llm_chain = prompt | llm_with_tools

# Adding explicit tool call check and logging/debugging info
@chain
def tool_chain(user_input: str, config: RunnableConfig):
    # Log: Tool called
    st.write("Tool called. Running Tavily search...")
    
    # Dynamically create the user input part of the prompt (article content)
    USER_PROMPT = f"""
    Use the Tavily search tool to find the latest information to complete your task.
    <article>{user_input}</article>
    """
    
    # Pass the dynamically generated user prompt into the input
    input_ = {"user_input": USER_PROMPT}
    
    # Initial invocation of the model with the prompt
    ai_msg = llm_chain.invoke(input_, config=config)
    
    # Log: Check if tool_calls exist in the message
    if ai_msg.tool_calls:
        st.write("Tool call detected. Fetching search results...")
        
        # Process tool results and inspect structure of tool_msgs
        tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
        
        # Log the structure of tool_msgs
        st.write("Tool message structure: ", tool_msgs)
        
        # Assuming `tool_msgs` has a `content` attribute or similar for URLs
        # Modify according to actual structure
        search_results = [msg.content['url'] for msg in tool_msgs if 'url' in msg.content]
        
        # Log: Show URLs searched
        st.write("Search URLs returned by Tavily:")
        for url in search_results:
            st.write(f"- {url}")
        
        # Re-invoke the chain with tool results
        st.write("Data processed. Sending to Claude model for final output...")
        final_output = llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
        
        # Log: Output from the model
        st.write("Model output received:")
        st.write(final_output.content)
        
        return final_output
    else:
        st.write("No tool calls detected, returning Claude model response directly.")
        return ai_msg

# Streamlit UI
st.title("Content Validator 0.2")

# User input for search query
st.subheader("Search Query")
user_input = st.text_area("Enter your query here:", height=200)

if st.button("Run Search"):
    if user_input:
        st.session_state.user_input = user_input
        with st.spinner("Running search..."):
            inputs = {
                "user_input": st.session_state.user_input
            }
            response = tool_chain.invoke(inputs, RunnableConfig())
            
            # Display the result
            if isinstance(response, AIMessage):
                st.write("Claude's Response:")
                st.write(response.content)
            else:
                st.write("Search Result:")
                st.write(response.content)
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "ai", "content": response.content})
    else:
        st.error("Please enter a search query.")

# Display analysis history
st.subheader("Search History")
for message in st.session_state.chat_history:
    st.write(f"{message['role'].capitalize()}:")
    st.write(message.get('content', ''))
    st.write("---")

# Clear history button
if st.button("Clear History"):
    st.session_state.chat_history = []
    st.session_state.user_input = ""
    st.success("History cleared.")
