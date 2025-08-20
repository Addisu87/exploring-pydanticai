import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn
from weather_service import WeatherService

weather_agent = Agent(
    "openai:gpt-4o",
    deps_type=WeatherService,
    system_prompt="Providing a weather forecast at the locations the user provides.",
)

@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date
) -> str:
    if forecast_date < date.today():
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(
    user_prompts: list[tuple[str, int]],
    conn: DatabaseConn
    ):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
