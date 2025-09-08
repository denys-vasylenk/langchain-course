from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schema import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_veraibles=[
        "tools",
        "tool_names",
        "input",
        "format_instructions",
        "agent_scratchpad",
    ],
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)

exctract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | exctract_output | parse_output


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job position for an AI engeneer using LangChain in the Kyiv on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
