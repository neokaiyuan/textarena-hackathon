from dotenv import load_dotenv

load_dotenv()

import textarena as ta

# Initialize agents
agents = {
    0: ta.agents.AWSBedrockAgent(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2"
    ),
    1: ta.agents.AWSBedrockAgent(
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0", region_name="us-west-2"
    ),
}

# Initialize environment from subset and wrap it
env = ta.make(env_id="Poker-v0")
env = ta.wrappers.LLMObservationWrapper(env=env)
env = ta.wrappers.SimpleRenderWrapper(
    env=env,
    player_names={0: "sonnet", 1: "haiku"},
)

env.reset(num_players=len(agents))
done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, info = env.step(action=action)
rewards = env.close()
