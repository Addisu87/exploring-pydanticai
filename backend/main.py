import asyncio

import logfire
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from typing_extensions import TypedDict

logfire.configure()
logfire.instrument_httpx()


logfire.info("Hello, {name}!", name="world")


class NeverResultType(TypedDict):
    """Never ever coerce data to this type."""

    never_use_this: str


agent = Agent(
    "openai:gpt-4o",
    retries=3,
    result_type=NeverResultType,
    system_prompt="Any time you get a response, call the `infinite_retry_tool` to produce another response.",
    model_settings={"temperature": 0.0},
)


@agent.tool_plain(retries=5)
def infinite_retry_tool() -> int:
    raise ModelRetry("Please try again.")


try:
    result_sync = agent.run_sync(
        "Begin infinite retry loop!",
        usage_limits=UsageLimits(request_limit=3),
    )
except UsageLimitExceeded as e:
    print(e)
    # > The next request would exceed the request_limit of 3


async def main():
    result = await agent.run("What is the capital of France?")
    print(result.data)
    # > Paris

    async with agent.run_stream("What is the capital of Germany?") as response:
        print(response.get_data)
        # > Berlin


if __name__ == "__main___":
    asyncio.run(main())
