from dataclasses import dataclass

import logfire
from core.config import settings
from db.base import DatabaseConn
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic()


model = OpenAIModel(
    "deepseek-chat",
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.BASE_URL,
)


# SupportDependencies is used to pass data, connections, and logic into that will be needed when running system prompt and tool functions.
# Dependency injection provides a type-safe way to customize the behavior of your agents.
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# This pydantic model defines the structure of the result returned by the agent.
class SupportResult(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: int = Field(description="Whether to block the customer's card")
    risk: int = Field(description="Risk level of query", ge=0, le=10)


# This agent will act as first-tier support in a bank.
# Agents are generic in the type of dependencies they accept and the type of result they return.
# In this case, the support agent has type `Agent[SupportDependencies, SupportResult]`.
support_agent = Agent(
    model,
    deps_type=SupportDependencies,
    # The response from the agent will, be guaranteed to be a SupportResult,
    # if validation fails the agent is prompted to try again.
    result_type=SupportResult,
    system_prompt=(
        "You are a support agent in our bank, give the "
        "customer support and judge the risk level of their query."
    ),
)


# Dynamic system prompts can make use of dependency injection.
# Dependencies are carried via the `RunContext` argument, which is parametrized with the `deps_type` from above.
# If the type annotation here is wrong, static type checkers will catch it.
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


# `tool` let you register functions which the LLM may call while responding to a user.
# Again, dependencies are carried via `RunContext`, any other arguments become the tool schema passed to the LLM.
# Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
    """Returns the customer's current account balance."""
    # The docstring of a tool is also passed to the LLM as the description of the tool.
    # Parameter descriptions are extracted from the docstring and added to the parameter schema sent to the LLM.
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f"${balance:.2f}"


...  # In a real use case, you'd add more tools and a longer system prompt


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    # Run the agent asynchronously, conducting a conversation with the LLM until a final response is reached.
    # Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve a result.
    result = await support_agent.run("What is my balance?", deps=deps)
    # The result will be validated with Pydantic to guarantee it is a `SupportResult`, since the agent is generic,
    # it'll also be typed as a `SupportResult` to aid with static type checking.
    print(result.data)
    """support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run("I just lost my card!", deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
