from __future__ import annotations

from enum import Enum

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import Tool, ToolDefinition


# Define an Enum for dependency types
class DepsType(Enum):
    HUMAN = "human"
    MACHINE = "machine"


def greet(name: str) -> str:
    return f"hello {name}"


# Update prepare_greet to use the Enum's value
async def prepare_greet(
    ctx: RunContext[DepsType],
    tool_def: ToolDefinition,
) -> ToolDefinition | None:
    d = f"Name of the {ctx.deps} to greet."
    tool_def.parameters_json_schema["properties"]["name"]["description"] = d
    return tool_def


greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
# Pass the Enum class as deps_type
agent = Agent(
    test_model,
    tools=[greet_tool],
    deps_type=DepsType,
)

result = agent.run_sync("testing...", deps=DepsType.HUMAN)
print(result.data)
# > {"greet":"hello a"}
print(test_model.agent_model_function_tools)
"""
[
    ToolDefinition(
        name='greet',
        description='',
        parameters_json_schema={
            'properties': {
                'name': {
                    'title': 'Name',
                    'type': 'string',
                    'description': 'Name of the human to greet.',
                }
            },
            'required': ['name'],
            'type': 'object',
            'additionalProperties': False,
        },
        outer_typed_dict_key=None,
    )
]
"""
