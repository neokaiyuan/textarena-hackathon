import re
import pprint
import time

from dotenv import load_dotenv
load_dotenv()

import textarena as ta
import smithery
import mcp
import os
import json

from textarena.core import Agent
import textarena as ta
import asyncio
from typing import Optional

model_name = "JUST FOR FUN"
model_description = "JUST FOR FUN"
email = "varicklsr@gmail.com"

os.environ["ANTHROPIC_API_KEY"] = "sk-ant"
SYSTEM_PROMPT = "You are Warren Buffet. Make sure you read the game instructions carefully, and always follow the required format."
negotiationAgent = ta.agents.AnthropicAgent(model_name="claude-3-7-sonnet-20250219", system_prompt=SYSTEM_PROMPT) 

for _ in range(10):
    try:
        # Initialize agent
        env = ta.make_online(
            env_id=["SimpleNegotiation-v0"],
            model_name=model_name,
            model_description=model_description,
            email=email
        )
        env = ta.wrappers.LLMObservationWrapper(env=env)

        env.reset(num_players=1)

        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = negotiationAgent(observation)
            done, info = env.step(action=action)
        env.close()
        print(info)
        print(f"DONE - sleeping")
        time.sleep(1)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        time.sleep(1)

# negotiation_agent = WarrenBuffetAgent(model_name="claude-3-7-sonnet-20250219")

# env = ta.make_online(
#     env_id=["SimpleNegotiation-v0"],
#     model_name=model_name,
#     model_description=model_description,
#     email=email
# )
# env = ta.wrappers.LLMObservationWrapper(env=env)

# env.reset(num_players=1)

# done = False
# while not done:
#     player_id, observation = env.get_observation()
#     action = agent(observation)
#     done, info = env.step(action=action)

# rewards = env.close()