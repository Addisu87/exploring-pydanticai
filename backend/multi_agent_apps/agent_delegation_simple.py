from dataclasses import dataclass
import httpx

from pydantic_ai import Agent, RunContext
# from pydantic_ai.usage import UsageLimits

@dataclass
class ClientAndKey:
    http_client: httpx.AsyncClient 
    api_key: str 
    
joke_selection_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'Use the `joke_factory` tool to generate some jokes on the given subject, '
        'then choose the best. You must return just a single joke.'
    ),
)

joke_generation_agent = Agent(
    'gemini-1.5-flash',
    deps_type=ClientAndKey,
    output_type=list[str],
    system_prompt=(
        'Use the "get_jokes" tool to get some jokes on the given subject, '
        'then extract each joke into a list.'
    )
)

@joke_selection_agent.tool 
async def joke_factory(ctx: RunContext[ClientAndKey], count: int) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com',
        params={'count': count},
        headers={"Authorization": f'Bearer {ctx.deps.api_key}'},
        # usage = ctx.usage,
    )
    response.raise_for_status()
    return response.text

# result = joke_selection_agent.run_sync(
#     "Tell me a joke.",
#     usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300)
    
# )

# print(result.output)
# #> Did you hear about the toothpaste scandal? They called it Colgate.
# print(result.usage())
# #> RunUsage(input_tokens=204, output_tokens=24, requests=3)

async def main():
    async with httpx.AsyncClient() as client: 
        deps = ClientAndKey(client, 'foobar')
        result = await joke_selection_agent.run('Tell me a joke.', deps=deps)
        print(result.output)
        #> Did you hear about the toothpaste scandal? They called it Colgate.
        print(result.usage())
        #> RunUsage(input_tokens=309, output_tokens=32, requests=4)