import asyncio
import textarena as ta
from mcp_agent import MCPAgent
from dotenv import load_dotenv

load_dotenv()


model_name = "[Just For Fun] Kai"
model_description = "3.7"
email = "kaiyuan.neo@gmail.com"

# Initialize agents
agent = MCPAgent(model_name="claude-3-7-sonnet-latest")


# Initialize environment from subset and wrap it
env = ta.make_online(
    env_id=["Poker-v0"],
    model_name=model_name,
    model_description=model_description,
    email=email,
)
env = ta.wrappers.LLMObservationWrapper(env=env)

env.reset(num_players=1)
done = False
while not done:
    player_id, observation = env.get_observation()
    action = asyncio.get_event_loop().run_until_complete(agent(observation))
    done, info = env.step(action=action)
rewards = env.close()
