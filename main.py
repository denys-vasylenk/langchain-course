from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def calc_tokens_OpenAI(metadata, input_price_per1K, output_price_per1K):
  input_price = input_price_per1K * (metadata["token_usage"]['prompt_tokens'] / 1000)
  output_price = output_price_per1K * (metadata["token_usage"]['completion_tokens'] / 1000)
  return print(f"Input token price: {input_price} $\nOutput token price:{output_price} $\nTotal:             {input_price+output_price} $")

def calc_tokens_llama(metadata, input_price_per1K, output_price_per1K):
  input_price = input_price_per1K * (metadata["prompt_eval_count"] / 1000)
  output_price = output_price_per1K * (metadata["eval_count"] / 1000)
  return print(f"Input token price: {round(input_price,7)} $\nOutput token price:{round(output_price,7)} $\nTotal:             {round(input_price+output_price,7)} $")

def main():
    print("langchain course")
    # information = """
    # Elon Reeve Musk born June 28, 1971 is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.
    # Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.
    # In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research but later left; growing discontent with the organization's direction and their leadership in the AI boom in the 2020s led him to establish xAI. In 2022, he acquired the social network Twitter, implementing significant changes and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017.
    # Musk was the largest donor in the 2024 U.S. presidential election, and is a supporter of global far-right figures, causes, and political parties. In early 2025, he served as senior advisor to United States president Donald Trump and as the de facto head of DOGE. After a public feud with Trump, Musk left the Trump administration and announced he was creating his own political party, the America Party.
    # Musk's political activities, views, and statements have made him a polarizing figure, especially following the COVID-19 pandemic. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories, and affirming antisemitic, racist, and transphobic comments. His acquisition of Twitter was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. His role in the second Trump administration attracted public backlash, particularly in response to DOGE.
    # """

    # summary_template = """
    # Given the information about the person I want you to create:
    # 1. A short summary
    # 2. Two interesting facts about them

    # Infromation: {information}
    # """

    with open("/Users/me/Desktop/langchain-course/client_input.txt", "r", encoding="utf-8") as f:
        client_input = f.read()

    with open("/Users/me/Desktop/langchain-course/prompt1.txt", "r", encoding="utf-8") as f:
        prompt1 = f.read()

    summary_prompt_template = PromptTemplate(
        input_variables=["client_input"], template=prompt1
    )


    temp = 0.2
    models = [ "gpt-3.5-turbo", "gemma3:270m"]
    model = models[1]
    input_token_price_per1K = 0.0015
    output_token_price_per1K = 0.002


    if "gpt-" in model:
        llm = ChatOpenAI(temperature=temp, model=model)
        token_calc = calc_tokens_OpenAI
    else:
        llm = ChatOllama(temperature=temp, model=model)
        token_calc = calc_tokens_llama
    
            
    chain = summary_prompt_template | llm

    response = chain.invoke(input ={"client_input":client_input})
    print(f"Output:\n{response.content}")
    print("Price:")
    print(token_calc(response.response_metadata, input_token_price_per1K, output_token_price_per1K))
    # print(response.response_metadata)


if __name__ == "__main__":
    main()
