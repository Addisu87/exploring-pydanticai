# TestModel
from datetime import timezone

import pytest
from db.fake_database import DatabaseConn
from dirty_equals import IsNow
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from weather_agent.weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.asyncio
models.ALLOW_MODEL_REQUESTS = False


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
                    tool_call_id=None,
                )
            ],
            model_name="test",
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="weather_forecast",
                    content="Sunny with a chance of rain",
                    tool_call_id=None,
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
            model_name="test",
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]
