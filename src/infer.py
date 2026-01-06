# main.py

import os
import torch
from dotenv import load_dotenv
from pprint import pprint

from models import Deps
from setup import load_deps
from agent import build_graph

# Load environment variables
load_dotenv()

# Input query
# query = "Someone swallowed a cleaning chemical by accident. What should I do immediately?"
query = "My friend is choking and cant speak, how can I help them?"

if __name__ == "__main__":
    # Load dependencies (models, retriever, etc.)
    deps = load_deps()

    # Build the agent graph
    agent = build_graph(deps)

    state = {}   # <-- keep it OUTSIDE the loop

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("exit", "quit"):
            break

        state["question"] = user_text

        state = agent.invoke(state)

        print("\nAgent:", state.get("final_answer", ""), "\n")

        # Debug
        # print("DEBUG:", {k: state.get(k) for k in ["pending_emergency_lookup","wants_emergency_number","location"]})


