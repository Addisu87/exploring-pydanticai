# Results
from typing import Union

from fake_database import DatabaseConn, QueryError
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext


class Success(BaseModel):
    sql_query: str


class InvalidRequest(BaseModel):
    error_message: str


Response = Union[Success, InvalidRequest]
agent: Agent[DatabaseConn, Response] = Agent(
    "openai:gpt-4o-mini",
    result_type=Response,  # type: ignore
    system_prompt="Generate PostgreSQL flavored SQL queries based on the given input.",
)


@agent.result_validator
async def validate_result(ctx: RunContext[DatabaseConn], result: Response) -> Response:
    if isinstance(result, InvalidRequest):
        return result
    try:
        await ctx.deps.execute(f"EXPLAIN {result.sql_query}")
    except QueryError as e:
        raise ModelRetry(f"Invalid query: {e}") from e
    else:
        return result


result = agent.run_sync(
    "get me users who were last active yesterday.", deps=DatabaseConn()
)
print(result.data)

# > sql_query="SELECT * FROM users WHERE last_active::date=today() - interval 1 day"
