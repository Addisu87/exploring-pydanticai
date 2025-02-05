import asyncio

from ai_q_and_a_graph import Answer, Ask, QuestionState, question_graph
from pydantic_graph import End, HistoryStep
from rich.prompt import Prompt


async def main():
    state = QuestionState()
    node = Ask()
    history = list[HistoryStep[QuestionState]] = []  # type: ignore

    while True:
        node = await question_graph.next(node, history, state=state)  # type: ignore
        if isinstance(node, Answer):
            node.answer = Prompt.ask(node.question)
        elif isinstance(node, End):
            print(f"Correct answer! {node.data}")
            # > Correct answer! Well done, 1 + 1 = 2
            print([e.data_snapshot() for e in history])
            """
            [
                Ask(),
                Answer(question='What is the capital of France?', answer='Vichy'),
                Evaluate(answer='Vichy'),
                Reprimand(comment='Vichy is no longer the capital of France.'),
                Ask(),
                Answer(question='what is 1 + 1?', answer='2'),
                Evaluate(answer='2'),
            ]
            """
            return
    # otherwise just continue


if __name__ == "__main___":
    asyncio.run(main())
