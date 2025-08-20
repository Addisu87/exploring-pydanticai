from pydantic_ai import Agent

agent = Agent("openai:gpt-4o", system_prompt="Be a helpful assistant.")

result1 = agent.run_sync("Tell me a joke.")
print(result1.output)

# > Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync(
            "Explain?",
            model='google-gla:gemini-1.5-pro',
            message_history=result1.new_messages()
)
print(result2.output)
# > This is an excellent joke invented by Samuel Colvin, it needs no explanation.

# all messages from the run
print(result2.all_messages())

"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ],
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.',
            )
        ],
        usage=RequestUsage(input_tokens=0, output_tokens=12),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ],
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
            )
        ],
        usage=RequestUsage(input_tokens=61, output_tokens=26),
        model_name='gemini-1.5-pro',
        timestamp=datetime.datetime(...),
    ),
]
"""
