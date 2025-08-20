from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

# Use a cheaper model to summarize old messages.
summarize_agent = Agent(
    'openai:gpt-4o-mini',
    instructions="""
        Summarize this conversation, omitting small talk and unrelated topics.
        Focus on the technical discussion and next steps.
    """,
)


async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Summarize the oldest 10 messages
    if len(messages) > 10:
        oldest_messages = messages[:10]
        summary = await summarize_agent.run(message_history=oldest_messages)
        # Return the last message and the summary
        return summary.new_messages() + messages[-1:]

    return messages


agent = Agent('openai:gpt-4o', history_processors=[summarize_old_messages]) 