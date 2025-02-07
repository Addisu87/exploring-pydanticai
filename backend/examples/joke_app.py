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

    async def system_prompt_factory(self) -> str:
        response = await self.http_client.get("https://example.com")
        response.raise_for_status()
        return f"Prompt: {response.text}"


joke_agent = Agent(
    "openai:gpt-4o",
    deps_type=MyDeps,
)


# System prompt functions
@joke_agent.system_prompt
async def get_system_prompt(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.system_prompt_factory()


# Function tools
@joke_agent.tool
async def get_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
    response = await ctx.deps.http_client.get(
        "https://example.com#jokes",
        params={"subject": subject},
        headers={"Authorization": f"Bearer {ctx.deps.api_key}"},
    )
    response.raise_for_status()
    return response.text


# Result validations
@joke_agent.result_validator
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


async def application_code(prompt: str) -> str:
    # now deep within application code we call our agent
    async with httpx.AsyncClient() as client:
        app_deps = MyDeps("foobar", client)
        result = await joke_agent.run(prompt, deps=app_deps)
    return result.data
