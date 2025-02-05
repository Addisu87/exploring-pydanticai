from pydantic_ai import Agent

agent = Agent("openai:gpt-4o", system_prompt="Be a helpful assistant.")

result1 = agent.run_sync("Tell me a joke.")
print(result1.data)

# > Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync("Explain?", message_history=result1.new_messages())
print(result2.data)
# > This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())

"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                dynamic_ref=None,
                part_kind='system-prompt',
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            ),
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
                part_kind='text',
            )
        ],
        model_name='function:model_logic',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
