import asyncio
from dotenv import load_dotenv

from agents.MCPAgent import MCPAgent
from game_envs import GameEnv
load_dotenv()

import textarena as ta

model_name = "[Just for Fun] MCP Agent"
model_description = "Just for Fun"
email = "jerome.chuame@gmail.com"

MAX_ROUNDS = 10
STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."

async def main():
    # Initialize agents
    agents = {
        0: MCPAgent(model_name="claude-3-7-sonnet-20250219"),
        1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
    }

    # Initialize environment from subset and wrap it
    env = ta.make(env_id=GameEnv.SPELLING_BEE.value)
    env = ta.wrappers.LLMObservationWrapper(env=env)
    env = ta.wrappers.SimpleRenderWrapper(
        env=env,
        player_names={0: "MCPAgent", 1: "claude-3.5-haiku"},
    )   

    env.reset(num_players=len(agents)) 

    done = False

    match_round = 0
    while not done or match_round <= MAX_ROUNDS:
        player_id, observation = env.get_observation()

        agent = agents[player_id]
        if asyncio.iscoroutinefunction(getattr(agent, '__call__', None)):
            # For async agents
            action = await agent(observation)
        else:
            # For synchronous agents
            action = agent(observation)

        done, info = env.step(action=action)
        match_round += 1

    rewards = env.close()
    return rewards

if __name__ == "__main__":
    rewards = asyncio.run(main())
    print(f"Game complete. Rewards: {rewards}")