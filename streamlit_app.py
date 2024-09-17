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
    api_key=claude_api_key,
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
    # Dynamically create the user input part of the prompt (article content)
    note = '''Your output should always be in markdown. Never use $ in your output, only "\\$'''

    USER_PROMPT = f"""
    Use the Tavily search tool to find the latest information to complete your task.
    You are currently working at Singsaver, a financial product aggregator in Singapore.
    Articles you write may have product placements such as credit cards, bank accounts, loan products, etc.
    A new writer in your team submitted an article to review, and you heard from colleagues that the new writer tends to get the product details wrong.
    Sometimes the new writer writes the wrong interest details, or the wrong miles, or even an old card name—you caught them previously mentioning a product that was discontinued last year.
    Review their new article below (marked by the <article></article> XML tags) diligently, taking care to go through your review process three times at least: 
    1. Extract all the products mentioned in the article.
    2. List out to yourself all the details about those products one by one.
    3. Browse the internet to validate every detail about those products mentioned in the article.
    4. If the product is relevant and up to date, move on. If there is a mistake, however, highlight it and return a short description of the correct product details.
    5. Review the article again, this time validating there are no mentions of Singsaver's competitors, such as MoneySmart.

    <article>
    {user_input}
    </article>

    {note}
    """
    
    # Pass the dynamically generated user prompt into the input
    input_ = {"user_input": USER_PROMPT}
    
    # Initial invocation of the model with the prompt
    ai_msg = llm_chain.invoke(input_, config=config)
    
    # Check if tool_calls exist in the message
    if ai_msg.tool_calls:
        tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
        
        # Re-invoke the chain with tool results
        return llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
    else:
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
                st.markdown("Claude's Response:")
                st.markdown(response.content)
            else:
                st.markdown("Search Result:")
                st.markdown(response.content)
            
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
