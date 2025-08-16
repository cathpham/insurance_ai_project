## Using General OpenAI Client
from openai import OpenAI

# client = OpenAI()  ## Assumes OPENAI_API_KEY is set

# client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = os.environ.get("NVIDIA_API_KEY", "")
# )

client = OpenAI(
    base_url = "http://llm_client:9000/v1",
    api_key = "I don't have one"
)

completion = client.chat.completions.create(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    # model="gpt-4-turbo-2024-04-09",
    messages=[{"role":"user","content":"Hello World"}],
    temperature=1,
    top_p=1,
    max_tokens=1024,
    stream=True,
)

## Streaming with Generator: Results come out as they're generated
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")