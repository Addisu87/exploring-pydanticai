from typing import Literal, Union
from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, RunContext 
from pydantic_ai.messages import ModelMessage 
from pydantic_ai.usage import RunUsage, UsageLimits 

class FlightDetail(BaseModel):
    flight_number: str 
    