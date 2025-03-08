import asyncio
import textarena as ta
from textarena.agents.basic_agents import AsyncAnthropicAgent
from mcp_agent import MCPAgent
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """
You are Warren Buffet. Make sure you read the game instructions carefully, and always follow the required format.
"""

# Initialize agents
agents = {
    0: MCPAgent(model_name="claude-3-7-sonnet-latest", system_prompt=SYSTEM_PROMPT),
    1: AsyncAnthropicAgent(model_name="claude-3-5-haiku-latest"),
}


# Initialize environment from subset and wrap it
env = ta.make(env_id=["SpellingBee-v0", "SimpleNegotiation-v0", "Poker-v0"])
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "sonnet", 1: "haiku"},
)


async def run_game():
    env.reset(num_players=len(agents))
    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = await agents[player_id](observation)
        done, info = env.step(action=action)
    rewards = env.close()
    print(rewards)


asyncio.run(run_game())
