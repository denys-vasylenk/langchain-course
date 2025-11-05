from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools import Tool
from typing import Any

load_dotenv()

instructions = """
You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
You have qrcode package installed
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
"""


def main():
    print("Start...")
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model= "gpt-4o"),
        tools=[PythonREPLTool()]
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=[PythonREPLTool()], verbose=True)

    csv_agent_executor : AgentExecutor = create_csv_agent(
        llm = ChatOpenAI(temperature=0, model= "gpt-4o", verbose=True),
        path="episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )

    #### Router Agent
    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name='Python Agent',
            func=python_agent_executor_wrapper,
            description="""
            Useful when you need to transform natural language to python and execute the python code,
            returns the result of code execution.
            DOES NOT ACEPT CODE AS INPUT.
            """
        ),
        Tool(
            name='CSV Agent',
            func=csv_agent_executor.invoke,
            description="""
            Useful when you need to answer questions over episode_info.csv file.
            Takes an inut the entire question and returns the answer after running pandas calculations.
            """
        )
    ]
    
    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model= "gpt-4o"),
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)
    print(
        grand_agent_executor.invoke({
            "input": "Generate and save 5 QR codes in the current directory that lead to this link: https://www.linkedin.com/in/denys-vasylenko/"
        })
    )


if __name__ == "__main__":
    main()
