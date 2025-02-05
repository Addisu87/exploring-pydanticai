from __future__ import annotations as _annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, Graph, GraphRunContext


@dataclass
class User:
    name: str
    email: EmailStr
    interests: list[str]


@dataclass
class Email:
    subject: str
    body: str


@dataclass
class State:
    user: User
    write_agent_messages: list[ModelMessage] = field(default_factory=list)


email_write_agent = Agent(
    "google-vertex:gemini-1.5-pro",
    result_type=Email,
    system_prompt="Write a welcome email to our tech blog.",
)


@dataclass
class WriteEmail(BaseModel[State]):
    email_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[State]) -> Feedback:
        if self.email_feedback:
            prompt = (
                f"Rewrite the email for the user:\n"
                f"{format_as_xml(ctx.state.user)}\n"
                f"Feedback: {self.email_feedback}"
            )
        else:
            prompt = (
                f"Write a welcome email for the user:\n {format_as_xml(ctx.state.user)}"
            )

        result = await email_write_agent.run(
            prompt,
            message_history=ctx.state.write_agent_messages,
        )
        ctx.state.write_agent_messages += result.all_messages()
        return Feedback(result.data)


class EmailRequiresWrite(BaseModel):
    feedback: str


class EmailOk(BaseModel):
    pass


feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
    "openai:gpt-4o",
    result_type=EmailRequiresWrite | EmailOk,  # type: ignore
    system_prompt=(
        "Review the email and provide feedback, email must reference the users specific interests."
    ),
)


@dataclass
class Feedback(BaseNode[State, None, Email]):
    email: Email

    async def run(self, ctx: GraphRunContext[State]) -> WriteEmail | End[Email]:
        prompt = format_as_xml({"user": ctx.state.user, "email": self.email})
        result = await feedback_agent.run(prompt)
        if isinstance(result.data, EmailRequiresWrite):
            return WriteEmail(email_feedback=result.data.feedback)
        else:
            return End(self.email)


async def main():
    user = User(
        name="John Doe",
        email="john.joe@example.com",
        interests=["Haskel", "Lisp", "Fortran"],
    )

    state = State(user)
    feedback_graph = Graph(nodes=(WriteEmail, Feedback))  # type: ignore
    email, _ = await feedback_graph.run(WriteEmail(), state=state)  # type: ignore
    print(email)
    """Email(
        subject="Welcome to our tech blog!",
        body="Hello John, Welcome to our tech blog! ..."
    )"""
