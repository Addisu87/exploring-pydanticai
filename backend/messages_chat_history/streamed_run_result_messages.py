from pydantic_ai import Agent

agent = Agent("openai:gpt-4o", system_prompt = "Be a  helpful assistant.")


async def main():
    async with agent.run_stream('Tell me a joke.') as result: 
        # incomplete messages before the stream finishes 
        print(result.all_messages())
        
        """
        [
            ModelRequest(
                parts = [
                    SystemPromptPart(
                        content= 'Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            )
        ]
        """
        
        async for text in result.stream_text():
            print(text)
            #> Did you hear
            #> Did you hear about the toothpaste
            #> Did you hear about the toothpaste scandal? They called
            #> Did you hear about the toothpaste scandal? They called it Colgate.
        
        # complete messages once the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts =[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            )
            ModelResponse (
                parts =[
                    TestPart(
                        content='Did you hear about the toothpaste scandal? They called it Collgate.',
                    ),
                ],
                
                usage=RequestUsage(input_tokens=50, output_tokens=12),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            ),
        ]
        """