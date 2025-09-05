from dotenv import load_dotenv

load_dotenv()
from langsmith import Client

client = Client()


run_id = "0bafaadc-9d89-419d-b435-64e448c85067"
run = client.read_run(run_id)

print(run.id)
print(run.name)
print(run.total_tokens)
print(run.prompt_tokens)
print(run.completion_tokens)
print(run.total_cost)
