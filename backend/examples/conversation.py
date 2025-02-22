from pydantic_ai import Agent

agent = Agent("openai:gpt-4o")

# First run
result1 = agent.run_sync("Who was Albert Einstein?")
print(result1.data)
# > Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.

# Second run, passing previous message
result2 = agent.run_sync(
    "What was his most famous equation?",
    message_history=result1.new_messages(),
)
print(result2.data)
# > Albert Einstein's most famous equation is E=mc^2.
