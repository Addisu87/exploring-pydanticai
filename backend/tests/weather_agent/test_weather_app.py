# Unit testing with TestModel
from datetime import timezone
import pytest

from dirty_equals import IsNow, IsStr

from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai.usage import RequestUsage

from backend.db.base import DatabaseConn
from weather_agent.weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.asyncio
models.ALLOW_MODEL_REQUESTS = False

# Overriding model via pytest fixtures
@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield


async def test_forecast(override_weather_agent: None):
    conn = DatabaseConn()
    user_id = 1

    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):
            prompt = "What will the weather be like in London on 2024-11-28?"
            await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast": "Sunny with a chance of rain"}'

    assert messages == [
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content="Providing a weather forecast at the locations the user provides.",
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content="What will the weather be like in London on 2024-11-28?",
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="weather_forecast",
                    args={
                        "location": "a",
                        "forecast_date": "2024-01-01",
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=RequestUsage(
                input_tokens=71,
                output_tokens=7,
            ),
            model_name="test",
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="weather_forecast",
                    content="Sunny with a chance of rain",
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=RequestUsage(
                input_tokens=77,
                output_tokens=16,
                ),
            model_name="test",
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]
