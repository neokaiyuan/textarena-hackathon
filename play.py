import textarena as ta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
# Initialize agents with API key
agents = {
    0: ta.agents.AnthropicAgent(model_name="claude-3-5-haiku-latest"),
    1: ta.agents.AnthropicAgent(model_name="claude-3-7-sonnet-latest"),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="Poker-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    record_video=True,
    player_names={0: "claude-3.5-haiku", 1: "claude-3-7-sonnet-latest"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()