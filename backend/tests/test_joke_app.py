from httpx import AsyncClient

from backend.banking_agent.joke_app import MyDeps, application_code, joke_agent


class TestMyDeps(MyDeps):
    async def system_prompt_factory(self) -> str:
        return "test prompt"


async def test_application_code(client: AsyncClient):
    test_deps = TestMyDeps("test_key", client)
    with joke_agent.override(deps=test_deps):
        joke = await application_code("Tell me a joke.")
    assert joke.startswith("Did you hear about the toothpaste scandal?")
