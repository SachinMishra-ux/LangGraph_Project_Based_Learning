from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command
from amadeus import Client, ResponseError
import uuid
import os

class TripState(TypedDict):
    starting_point: Optional[str]
    destination: Optional[str]
    budget: Optional[str]
    duration: Optional[str]
    proposed_plan: Optional[str]
    satisfied_with_plan: Optional[bool]
    done: Optional[bool]
    selected_flight: Optional[str]
    flight_source: Optional[str]
    flight_destination: Optional[str]
    flight_date: Optional[str]
    return_trip: Optional[bool]
    available_flights: Optional[list]

amadeus = Client(
    client_id=os.getenv("AMADEUS_CLIENT_ID") or "YOUR_CLIENT_ID",
    client_secret=os.getenv("AMADEUS_CLIENT_SECRET") or "YOUR_CLIENT_SECRET"
)

from dotenv import load_dotenv
load_dotenv(override=True)

def get_groq_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2000
    )

llm = get_groq_llm()
FIELDS = ["starting_point", "destination", "budget", "duration"]
FIELD_QS = {
    "starting_point": "Ask the user for their trip starting location in a friendly way.",
    "destination": "Ask the user where they'd like to travel, naturally.",
    "budget": "Ask the user what budget they'd like to set for the trip, friendly and casual.",
    "duration": "Ask, in a conversational style, how many days they want their trip to last."
}

def generate_slot_question(slot_name):
    print("DEBUG: Entering generate_slot_question()")
    prompt = FIELD_QS[slot_name]
    response = llm.invoke(prompt).content.strip()
    print(f"DEBUG: Generated question for '{slot_name}': {response}")
    return response

def extract_slot_value(slot_name, user_input):
    print("DEBUG: Entering extract_slot_value()")
    prompt = (
        f"You are a smart assistant filling out a travel form.\n"
        f"The user was asked to specify their '{slot_name.replace('_', ' ')}'.\n"
        f"User replied: \"{user_input}\"\n"
        f"Extract and reply with ONLY the value for '{slot_name}'. If no clear value is found, reply 'None'."
    )
    response = llm.invoke(prompt).content.strip()
    if response.lower() == "none":
        response = None
    print(f"DEBUG: Extracted value for {slot_name}: '{response}'")
    return response

def extract_change_field(user_input: str) -> Optional[str]:
    """
    Detect which trip field (if any) the user wants to change, or if they want to proceed.
    Uses LLM for fuzzy detection, but is tolerant of natural human responses.
    """
    user_text = user_input.strip().lower()

    # --- Fast direct checks (no LLM call for simple confirmations) ---
    if any(word in user_text for word in ["proceed", "ok", "okay", "yes", "continue", "done", "confirm", "next", "go ahead", "move on"]):
        print("DEBUG: Direct 'proceed' detected without LLM.")
        return "proceed"

    # --- LLM fallback for detecting slot-change intent ---
    prompt = (
        "Decide which of these trip fields (if any) the user wants to update: "
        "starting_point, destination, budget, duration.\n"
        f"User said: \"{user_input}\"\n"
        "Reply ONLY with one of these: starting_point, destination, budget, duration, or 'proceed' if the user seems to approve or confirm."
    )

    try:
        result = llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f"DEBUG: extract_change_field LLM error: {e}")
        return None

    print(f"DEBUG: LLM slot change extraction: '{result}'")

    # --- Interpret LLM output robustly ---
    for f in FIELDS:
        if f in result:
            return f
    if any(word in result for word in ["proceed", "ok", "okay", "yes", "continue", "done", "confirm", "next"]):
        return "proceed"

    return None


def ask_starting_point(state: TripState) -> TripState:
    print("DEBUG: Entering ask_starting_point with state")
    if not state.get("starting_point"):
        q = generate_slot_question("starting_point")
        user_input = interrupt(q)
        value = extract_slot_value("starting_point", user_input)
        state["starting_point"] = value
    return state

def ask_destination(state: TripState) -> TripState:
    print("DEBUG: Entering ask_destination with state")
    if not state.get("destination"):
        q = generate_slot_question("destination")
        user_input = interrupt(q)
        value = extract_slot_value("destination", user_input)
        state["destination"] = value
    return state

def ask_budget(state: TripState) -> TripState:
    print("DEBUG: Entering ask_budget with state")
    if not state.get("budget"):
        q = generate_slot_question("budget")
        user_input = interrupt(q)
        value = extract_slot_value("budget", user_input)
        state["budget"] = value
    return state

def ask_duration(state: TripState) -> TripState:
    print("DEBUG: Entering ask_duration with state")
    if not state.get("duration"):
        q = generate_slot_question("duration")
        user_input = interrupt(q)
        value = extract_slot_value("duration", user_input)
        state["duration"] = value
    return state

def review_info(state: TripState) -> TripState:
    print("DEBUG: Entering review_info with state", state)

    # Step 1. Build and show summary
    summary = (
        f"Please review your trip details:\n"
        f"- Starting point: {state.get('starting_point')}\n"
        f"- Destination: {state.get('destination')}\n"
        f"- Budget: {state.get('budget')}\n"
        f"- Duration: {state.get('duration')}\n"
        "Reply 'proceed' to confirm, or say what you'd like to change (e.g. 'change budget to 50k')."
    )

    # Step 2. Get user response
    user_input = interrupt(summary)
    user_text = user_input.strip().lower()

    # Step 3. If user confirms, exit review
    if any(x in user_text for x in ["proceed", "ok", "okay", "yes", "continue", "done", "confirm", "next", "go ahead"]):
        print("DEBUG: Direct confirmation detected. Proceeding to next step.")
        return state  # ✅ confirmed

    # Step 4. Detect which field they want to change
    change_field = extract_change_field(user_input)
    print(f"DEBUG: Field to change detected: {change_field}")

    # Step 5. Handle field update
    if change_field in FIELDS:
        ask_prompt = f"Sure! What should be the new {change_field.replace('_',' ')}?"
        new_value_input = interrupt(ask_prompt)
        new_value = extract_slot_value(change_field, new_value_input)
        state[change_field] = new_value
        print(f"DEBUG: Updated {change_field} → {new_value}")

        # ✅ After updating, re-enter review to show updated summary
        return review_info(state)

    # Step 6. If unclear, gently re-ask
    reask = interrupt("Sorry, I didn’t catch that. Would you like to proceed or update something?")
    if any(x in reask.lower() for x in ["proceed", "ok", "yes", "continue"]):
        return state

    return review_info(state)  # ✅ loop back again




def review_router(state: TripState):
    print("DEBUG: In review_router with state", state)
    missing = [fld for fld in FIELDS if not state.get(fld)]
    print(f"DEBUG: In review_router, missing fields: {missing}")
    if len(missing) > 0:
        return "review_info"
    return "plan_trip"

def plan_trip(state: TripState) -> TripState:
    print("DEBUG: Entering plan_trip with state", state)
    state['satisfied_with_plan'] = None
    prompt = (
        f"Can you plan a fun, friendly, {state['duration']}-day itinerary from {state['starting_point']} "
        f"to {state['destination']} with a budget of {state['budget']}?"
    )
    result = llm.invoke(prompt)
    print(f"\nYour itinerary:\n{result.content}\n")
    state['proposed_plan'] = result.content
    return state

def plan_approval(state: TripState) -> TripState:
    print("DEBUG1: Entering plan_approval with state", state)
    review = (
        f"Here’s your trip plan:\n\n{state.get('proposed_plan','none')}\n\n"
        "Are you satisfied? Type 'yes', 'proceed', or 'ok' to book flights, or request changes."
    )
    user_input = interrupt(review)
    print(f"DEBUG2: User input in approval: {user_input!r}")
    change_field = extract_change_field(user_input)
    print(f"DEBUG: LLM approval extraction: {change_field!r}")
    if change_field in ("proceed", "yes", "ok"):
        state['satisfied_with_plan'] = True
    elif change_field in FIELDS:
        state[change_field] = None
        state['satisfied_with_plan'] = None
    else:
        # For anything else, also check if user's direct text is "yes"/etc for robustness
        if user_input.strip().lower() in ("proceed", "yes", "ok"):
            state['satisfied_with_plan'] = True
        else:
            state['satisfied_with_plan'] = None
    return state


def plan_approval_router(state: TripState):
    print(f"DEBUG: In plan_approval_router, satisfied_with_plan={state.get('satisfied_with_plan')}")
    if state.get("satisfied_with_plan"):
        print("DEBUG: Plan approved; moving to flight booking.")
        return "ask_flight_details"
    missing = [fld for fld in FIELDS if not state.get(fld)]
    if missing:
        print(f"DEBUG: Editing slot(s) {missing}; will re-review.")
        return "review_info"
    print("DEBUG: General feedback; re-planning.")
    return "plan_trip"




def ask_flight_details(state: TripState) -> TripState:
    print("DEBUG: Entering ask_flight_details with state", state)
    src = interrupt(f"To book your flight, please confirm your departure city (or edit): [{state['starting_point']}]")
    dst = interrupt(f"And your arrival city? [{state['destination']}]")
    date = interrupt("On what date would you like to depart? (YYYY-MM-DD)")
    ret_type = interrupt("Is this a one-way or round-trip? (Type 'one-way' or 'round-trip')")
    state['flight_source'] = src or state['starting_point']
    state['flight_destination'] = dst or state['destination']
    state['flight_date'] = date
    state['return_trip'] = "round" in ret_type.lower()
    return state

# def get_airport_code(city_name: str) -> str:
#     """Convert city name to IATA code using Amadeus API."""
#     try:
#         response = amadeus.reference_data.locations.get(
#             keyword=city_name,
#             subType='AIRPORT'
#         )
#         if response.data:
#             return response.data[0]['iataCode']
#     except ResponseError as e:
#         print(f"ERROR fetching airport code for {city_name}: {e}")
#     return None

FLIGHTAPI_KEY = ""  # move to env in prod

def get_airport_code(city_name: str) -> str | None:
    """
    Convert city name to IATA airport code using FlightAPI.
    """
    url = f"https://api.flightapi.io/iata/{FLIGHTAPI_KEY}"
    params = {
        "name": city_name,
        "type": "airport"
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        #print(payload)
        airports = payload.get("data", [])
        if airports:
            return airports[0].get("iata")

    except Exception as e:
        print(f"ERROR fetching airport code for {city_name}: {e}")

    return None

# def fetch_flights(state: dict) -> dict:
#     source_city = state.get('starting_point', '')
#     dest_city = state.get('destination', '')
#     travel_date = state.get('flight_date', '2025-11-01')  # default date fallback

#     print(f"DEBUG: Fetching flights from {source_city} → {dest_city} on {travel_date}")

#     origin_code = get_airport_code(source_city) or source_city.upper()
#     dest_code = get_airport_code(dest_city) or dest_city.upper()

#     print(f"DEBUG: Using airport codes {origin_code} → {dest_code}")

#     try:
#         response = amadeus.shopping.flight_offers_search.get(
#             originLocationCode=origin_code,
#             destinationLocationCode=dest_code,
#             departureDate=travel_date,
#             adults=1,
#             max=5
#         )
#         offers = response.data
#     except ResponseError as error:
#         print("ERROR fetching flights:", error)
#         offers = []

#     if not offers:
#         sel = interrupt(f"Sorry, couldn't find flights from {source_city} to {dest_city}. Try another route?")
#         state['selected_flight'] = None
#         return state

#     # Prepare display list
#     opts = []
#     for i, offer in enumerate(offers):
#         seg = offer["itineraries"][0]["segments"][0]
#         airline = seg["carrierCode"]
#         depart_time = seg["departure"]["at"]
#         arrive_time = seg["arrival"]["at"]
#         price = offer["price"]["total"]
#         currency = offer["price"]["currency"]
#         opts.append(f"{i+1}. {airline} | {depart_time} → {arrive_time} | {price} {currency}")

#     opts_text = "\n".join(opts)
#     sel = interrupt(f"Here are some real flights from {source_city} to {dest_city}:\n{opts_text}\nSelect a flight number:")

#     try:
#         ix = int(sel) - 1
#         if 0 <= ix < len(offers):
#             state['selected_flight'] = offers[ix]
#         else:
#             state['selected_flight'] = None
#     except ValueError:
#         state['selected_flight'] = None

#     return state

import requests

RAPIDAPI_KEY = ""

def fetch_flights(state: dict) -> dict:
    source_city = state.get("flight_source") or state.get("starting_point", "")
    dest_city = state.get("flight_destination") or state.get("destination", "")
    travel_date = state.get("flight_date", "2026-01-10")

    print(f"DEBUG: Fetching flights {source_city} → {dest_city} on {travel_date}")

    origin_code = get_airport_code(source_city) or source_city.upper()
    dest_code = get_airport_code(dest_city) or dest_city.upper()

    url = "https://flight-fare-search.p.rapidapi.com/v2/flights/"
    querystring = {
        "from": origin_code,
        "to": dest_code,
        "date": travel_date,
        "adult": "1",
        "type": "economy",
        "currency": "INR"
    }

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "flight-fare-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring, timeout=20)
    flights = response.json().get("results", [])[:4]

    if not flights:
        interrupt("Sorry, no flights found for this route.")
        return state

    # Show options
    options = []
    for i, flight in enumerate(flights):
        options.append(
            f"{i+1}. {flight['flight_code']} {flight['flight_name']} | "
            f"{flight['departureAirport']['time']} → {flight['arrivalAirport']['time']} | "
            f"{flight['totals']['total']} {flight['currency']}"
        )

    # ⬅️ THIS is the key line
    choice = interrupt(
        "Here are available flights:\n"
        + "\n".join(options)
        + "\n\nSelect a flight number:"
    )

    # ⬅️ This runs AFTER resume
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(flights):
            state["selected_flight"] = flights[idx]
            print("DEBUG: Flight selected successfully")
        else:
            state["selected_flight"] = None
    except ValueError:
        state["selected_flight"] = None

    return state




def booking_confirmation(state: dict) -> dict:
    flight = state.get("selected_flight")

    print("DEBUG booking_confirmation selected_flight:", state.get("selected_flight"))

    if not flight:
        interrupt("No flight was selected. Ending the booking flow.")
        state["done"] = True
        return state

    # 🔹 RapidAPI flight schema (matches fetch_flights)
    airline = flight.get("flight_name")
    flight_code = flight.get("flight_code")

    depart_airport = flight.get("departureAirport", {})
    arrive_airport = flight.get("arrivalAirport", {})

    depart_time = depart_airport.get("time")
    arrive_time = arrive_airport.get("time")

    price = flight.get("totals", {}).get("total")
    currency = flight.get("currency")

    confirm = interrupt(
        f"You selected **{airline} ({flight_code})**\n"
        f"✈️ {depart_time} → {arrive_time}\n"
        f"💰 Price: {price} {currency}\n\n"
        f"Would you like to confirm the booking? (yes/no)"
    )

    if confirm.strip().lower() in ("yes", "y"):
        print(f"✅ Booking confirmed for {airline} ({flight_code})!")
        state["done"] = True
        state["booking_status"] = "confirmed"
    else:
        print("❌ Booking cancelled by user.")
        state["done"] = True
        state["booking_status"] = "cancelled"

    return state

def build_workflow():

    graph = StateGraph(TripState)
    graph.add_node("ask_starting_point", ask_starting_point)
    graph.add_node("ask_destination", ask_destination)
    graph.add_node("ask_budget", ask_budget)
    graph.add_node("ask_duration", ask_duration)
    graph.add_node("review_info", review_info)
    graph.add_node("plan_trip", plan_trip)
    graph.add_node("plan_approval", plan_approval)
    graph.add_node("ask_flight_details", ask_flight_details)
    graph.add_node("fetch_flights", fetch_flights)
    graph.add_node("booking_confirmation", booking_confirmation)

    graph.set_entry_point("ask_starting_point")
    graph.add_edge("ask_starting_point", "ask_destination")
    graph.add_edge("ask_destination", "ask_budget")
    graph.add_edge("ask_budget", "ask_duration")
    graph.add_edge("ask_duration", "review_info")
    graph.add_conditional_edges(
        "review_info", review_router, {"review_info": "review_info", "plan_trip": "plan_trip"}
    )
    graph.add_edge("plan_trip", "plan_approval")
    graph.add_conditional_edges(
        "plan_approval", plan_approval_router,
        {
            "plan_trip": "plan_trip",
            "ask_flight_details": "ask_flight_details",
            "review_info": "review_info"
        }
    )
    graph.add_edge("ask_flight_details", "fetch_flights")
    graph.add_edge("fetch_flights", "booking_confirmation")
    graph.add_edge("booking_confirmation", END)

    return graph.compile(checkpointer=MemorySaver())