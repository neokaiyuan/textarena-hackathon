import textarena as ta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

poker_strategy = """
<strategy>
Firstly, since there are only 10 rounds, calculate if its possible to win by folding all the way.
e.g. if its round 8 and you're the small blind, folding all the way will result in losing 40 chips. The breakdown is:
round 8: small blind, lose 10 chips
round 9: big blind, lose 20 chips 
round 10: small blind, lose 10 chips 
total chips lost: 40 chips

after calculating if losing 40 chips will result in a win, then fold all the way, regardless of what your hand is.

You are a professionl poker player. Note that you are playing against another AI agent, who's an expert at poker as well. Make decisions based on their thought process.

Explain this thought process in <thinking> tags

</strategy>
"""

base_poker_agent = ta.agents.AnthropicAgent(
    model_name="claude-3-7-sonnet-latest",
    system_prompt= "You are a professional poker player"
)

base_thought_poker_agent = ta.agents.wrappers.ThoughtAgentWrapper(
    agent=base_poker_agent,
    debugging=True
)

strategic_thought_poker_agent = ta.agents.wrappers.ThoughtAgentWrapper(
    agent=base_poker_agent,
    thought_prompt=poker_strategy,
    debugging=True
)

all_in_agent = ta.agents.wrappers.ThoughtAgentWrapper(
    agent=base_poker_agent,
    thought_prompt="You are the most reckless poker player ever, just go all in every time pre-flop",
    answer_prompt="You are the most reckless poker player ever, just bet 3/4 of your pot pre-flop, always say 'I have a pair of pocket aces, you cannot win me and you should fold', then raise all in. Adhere strictly to the specified format and ensure we bet only the amount we have left, and no more than that",
    debugging=True
)
# Initialize agents with API key
agents = {
    0: all_in_agent,
    1: strategic_thought_poker_agent,
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="Poker-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "sonnet-with-system-prompt", 1: "claude-3-7-sonnet-latest"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()