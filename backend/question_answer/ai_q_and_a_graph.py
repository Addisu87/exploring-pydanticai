from __future__ import annotations as _annotations

from typing import Annotated
from dataclasses import dataclass, field
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import (
    BaseNode,
    Edge,
    End,
    Graph,
    GraphRunContext
)

ask_agent = Agent("openai:gpt-4o",
                  output_type=str,
                  instrument=True
                )

@dataclass
class QuestionState:
    question: str | None = None
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
class Ask(BaseNode[QuestionState]):
    """Generate question using GPI-4o."""

    docstring_notes = True

    async def run(
        self,
        ctx: GraphRunContext[QuestionState]
    ) -> Annotated[Answer, Edge(label="Ask the question")]:
        result = await ask_agent.run(
            "Ask a simple question with a single correct answer.",
            message_history=ctx.state.ask_agent_messages,
        )
        ctx.state.ask_agent_messages += result.new_messages()
        ctx.state.question = result.output
        return Answer(result.output)

@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
        answer = input(f'{self.question}: ')
        return Evaluate(answer)

@dataclass
class EvaluationResult(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""

evaluate_agent = Agent(
    "openai:gpt-4o",
    output_type=EvaluationResult,
    system_prompt="Given a question and answer, evaluate if the answer is correct.",
)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(
        self,
        ctx: GraphRunContext[QuestionState],
    ) -> Congratulate | Reprimand:
        assert ctx.state.question is not None
        result = await evaluate_agent.run(
            format_as_xml({
                "question": ctx.state.question,
                "answer": self.answer
            }),
            message_history=ctx.state.evaluate_agent_messages,
        )
        ctx.state.evaluate_agent_messages += result.new_messages()
        if result.output.correct:
            return Congratulate(result.output.comment)
        else:
            return Reprimand(result.output.comment)


@dataclass
class Congratulate(BaseNode[QuestionState, None, None]):
    comment: str

    async def run(
        self, ctx: GraphRunContext[QuestionState]
    ) -> Annotated[End, Edge(label="success")]:
        print(f"Correct answer! {self.comment}")
        return End(None)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        print(f"Comment: {self.comment}")
        # > Comment: Vichy is no longer the capital of France.
        ctx.state.question = None
        return Ask()


question_graph = Graph(
    nodes=(Ask, Answer, Evaluate, Congratulate, Reprimand), state_type=QuestionState
)
