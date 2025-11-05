from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are an agent that can use tools.
Always call only ONE tool per reasoning step.
After receiving the tool result, think again and decide the next tool or give the final answer.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state['messages']])
    return {"messages": [response]}

tool_node = ToolNode(tools)
