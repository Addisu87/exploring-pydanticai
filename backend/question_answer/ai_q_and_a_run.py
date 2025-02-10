import asyncio
import sys
from pathlib import Path

import logfire
from ai_q_and_a_graph import Answer, Ask, QuestionState, question_graph
from devtools import debug  # type: ignore
from pydantic_graph import End, HistoryStep


async def run_as_continuous():
    state = QuestionState()
    node = Ask()
    history: list[HistoryStep[QuestionState, None]] = []

    while logfire.span("run questions graph"):
        while True:
            node = await question_graph.next(node, history, state=state)  # type: ignore
            if isinstance(node, End):
                print(f"Correct answer! {node.data}")
                # > Correct answer! Well done, 1 + 1 = 2
                debug([e.data_snapshot() for e in history])
            elif isinstance(node, Answer):
                assert state.question
                node.answer = input(f"{state.question}")

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
    # otherwise just continue


async def run_as_cli(answer: str | None):
    history_file = Path("question_graph_history.json")
    history = (
        question_graph.load_history(history_file.read_bytes())
        if history_file.exists()
        else []
    )

    if history:
        last = history[-1]
        assert last.kind == "node", "expected last step to be a node"
        state = last.state
        assert answer is not None, "answer is required to continue from history"
        node = Answer(answer)
    else:
        state = QuestionState()
        node = Ask()
    debug(state, node)

    with logfire.span("run questions graph"):
        while True:
            node = await question_graph.next(node, history, state=state)
            if isinstance(node, End):
                debug([e.data_snapshot() for e in history])
                print("Finished!")
                break
            elif isinstance(node, Answer):
                print(state.question)
                break
            # otherwise just continue

    history_file.write_bytes(question_graph.dump_history(history, indent=2))


if __name__ == "__main__":
    try:
        sub_command = sys.argv[1]
        assert sub_command in ("continuous", "cli", "mermaid")
    except (IndexError, AssertionError):
        print(
            "Usage:\n"
            " uv run -m chat.question_graph mermaid\n"
            "or:\n"
            " uv run -m chat.question_graph continuous\n"
            "or:\n"
            " uv run -m chat.question_graph cli [answer]",
            file=sys.stderr,
        )
        sys.exit(1)

    if sub_command == "mermaid":
        print(question_graph.mermaid_code(start_node=Ask))
    elif sub_command == "continuous":
        asyncio.run(run_as_continuous())
    else:
        a = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(run_as_cli(a))
