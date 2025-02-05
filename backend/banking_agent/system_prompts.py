from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    "openai:gpt-4o",
    deps_type=str,
    system_prompt="Use the customer's name while replaying to them.",
)


@agent.system_prompt
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:
    return f"The date is {date.today()}."


result = agent.run_sync("What is the date?", deps="John")
print(result.data)
# > The date is 2025-03-24
