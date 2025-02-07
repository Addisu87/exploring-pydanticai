from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class User:
    name: str


agent = Agent(
    "test",
    deps_type=User,
    result_type=bool,
)


@agent.system_prompt  # type: ignore
async def add_user_name(ctx: RunContext[str]) -> str:
    return "The user's name is {ctx.deps}."


def foobar(x: bytes):
    pass


result = agent.run_sync("Does their name start with 'A'?", deps=User("Anne"))
foobar(bytes([result.data]))
