# Unit testing with FunctionModel

import re

import pytest

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from backend.db.base import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.asyncio
models.ALLOW_MODEL_REQUESTS = False


def call_weather_forecast(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r"\d{4}-\d{2}-\d{2}", user_prompt.content)
        assert m is not None
        args = {"location": "London", "forecast_date": m.group()}
        return ModelResponse(parts=[ToolCallPart("weather_forecast", args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == "tool-return"
        return ModelResponse(parts=[TextPart(f"The forecast is: {msg.content}")])


async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):
        prompt = "What will the weather be like in London on 2024-11-28?"
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast": "Sunny with a chance of rain"}'
