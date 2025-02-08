from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, ModelRetry, RunContext

# model = OpenAIModel(
#     "model_name",
#     base_url="https://<openai-compatible-api-endpoint>.com",
#     api_key="your-api-key",
# )


# Defining Dependencies
# Accessing dependencies
# Asynchronous vs. Synchronous Dependencies


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    "openai:gpt-4o",
    deps_type=MyDeps,
)


# System prompt functions
@agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    response = await ctx.deps.http_client.get("https://example.com")
    response.raise_for_status()
    return f"Prompt: {response.text}"


# Function tools
@agent.tool
async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
    response = await ctx.deps.http_client.get(
        "https://example.com#jokes",
        params={"subject": subject},
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
    )
    response.raise_for_status()
    return response.text


# Result validations
@agent.result_validator
async def validate_result(ctx: RunContext[MyDeps], final_response: str) -> str:
    response = await ctx.deps.http_client.post(
        "https://example.com#validate",
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
        params={"query": final_response},
    )

    if response.status_code == 400:
        raise ModelRetry(f"Invalid response: {response.text}")
    response.raise_for_status()
    return final_response


async def main():
    async with httpx.AsyncClient() as client:
        deps = MyDeps("foobar", client)
        result = await agent.run("Tell me a joke.", deps=deps)
        print(result.data)
        # > Did you hear about the tootpaste scandal? They called it Colgage.



