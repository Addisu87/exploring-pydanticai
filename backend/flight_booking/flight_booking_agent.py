import datetime
from dataclasses import dataclass
from typing import Literal

import logfire
from core.config import settings
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import Usage, UsageLimits
from rich.prompt import Prompt

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic()


model = OpenAIModel(
    "deepseek-chat",
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.BASE_URL,
)


# **Flight Data Models**
class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description="Three-letter airport code")
    destination: str = Field(description="Three-letter airport code")
    date: datetime.date


class NoFlightFound(BaseModel):
    """When no valid flight is found."""


@dataclass
class Deps:
    req_origin: str
    req_destination: str
    req_date: datetime.date
    available_flights: list[FlightDetails]


# **1. Flight Extraction Agent**
extraction_agent = Agent(
    model,
    result_type=list[FlightDetails],
    system_prompt="Extract all the flight details from the given text.",
)

# **2. Flight Search Agent**
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    model,
    result_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=2,
    system_prompt="Find the cheapest flight for the user based on extracted flights.",
)


@search_agent.tool
async def get_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Retrieve flights already extracted instead of re-calling LLM."""
    # We pass the usage to the search agent to requests within this agent are counted
    return ctx.deps.available_flights


@search_agent.result_validator
async def validate_result(
    ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Ensure the selected flight matches the requested criteria."""
    if isinstance(result, NoFlightFound):
        return result

    if (
        result.origin != ctx.deps.req_origin
        or result.destination != ctx.deps.req_destination
        or result.date != ctx.deps.req_date
    ):
        raise ModelRetry("Flight does not meet user constraints.")

    return result


# **3. Seat Selection Agent**
class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal["A", "B", "C", "D", "E", "F"]


class Failed(BaseModel):
    """Unable to extract a seat selection."""


seat_preference_agent = Agent[None, SeatPreference | Failed](
    model,
    result_type=SeatPreference | Failed,  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        "Seats A and F are window seats. "
        "Row 1 is the front row."
        "Row 1, 14, and 20 have extra leg room."
    ),
)


# **4. Booking Logic**
# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)


async def find_flight(deps: Deps, usage: Usage) -> FlightDetails | None:
    """Handles flight search and validation logic."""
    message_history: list[ModelMessage] | None = None

    result = await search_agent.run(
        f"Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}",
        deps=deps,
        usage=usage,
        message_history=message_history,
        usage_limits=usage_limits,
    )

    if isinstance(result.data, NoFlightFound):
        print("No suitable flight found.")
        return None

    return result.data


async def find_seat(usage: Usage) -> SeatPreference:
    """Handles seat selection with limited retries."""
    max_attempts = 3
    attempts = 0
    message_history: list[ModelMessage] | None = None

    while attempts < max_attempts:
        answer = Prompt.ask("What seat would you like? (e.g., 12A)")

        result = await seat_preference_agent.run(
            answer,
            usage=usage,
            usage_limits=usage_limits,
            message_history=message_history,
        )

        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print("Invalid seat selection. Try again.")
            message_history = result.all_messages()
            attempts += 1

    print("Max retries reached. Assigning default seat 10C.")
    return SeatPreference(row=10, seat="C")


async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    """Mock function to simulate purchasing a flight."""
    print(
        f"Purchasing flight {flight_details.flight_number} with seat {seat.row}{seat.seat}..."
    )


async def main():
    """Main flow for flight booking."""

    usage = Usage()

    # **Extract flights only once**
    result = await extraction_agent.run(flights_web_page, usage=usage)
    logfire.info("found {flight_count} flights", flight_count=len(result.data))
    available_flights = result.data

    if not available_flights:
        print("No flights available.")
        return

    deps = Deps(
        req_origin="SFO",
        req_destination="ANC",
        req_date=datetime.date(2025, 1, 10),
        available_flights=available_flights,
    )

    # **Find the best flight**
    flight = await find_flight(deps, usage)

    if not flight:
        return  # No flight found, exit

    print(f"Flight found: {flight}")

    action = Prompt.ask(
        "Do you want to buy this flight? (yes/no)",
        choices=["yes", "no"],
        show_choices=False,
    )

    if action == "yes":
        seat = await find_seat(usage)
        await buy_tickets(flight, seat)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
