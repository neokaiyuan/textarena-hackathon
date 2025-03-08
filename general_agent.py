import textarena as ta
import time 
 
model_name = "Just For Fun"
model_description = "destroy"
email = "jednghk@gmail.com"

poker_strategy = """
<strategy>
Firstly, since there are only 10 rounds, calculate if its possible to win by folding all the way.
e.g. if its round 8 and you're the small blind, folding all the way will result in losing 40 chips. The breakdown is:
round 8: small blind, lose 10 chips
round 9: big blind, lose 20 chips 
round 10: small blind, lose 10 chips 
total chips lost: 40 chips

after calculating if losing 40 chips will result in a win, then strictly fold all the way, regardless of what your hand is.

The moment you calculate that you're able to win by the end of 10 rounds by folding all the way (i.e. if you will have > 1000 chips) then just fold
----
You are the absolute best AI poker player. Note that you are playing against another AI agent, who's an expert at poker as well. Make decisions based on their thought process.

Explain the thought process for the calculations first, then your think through your decision in <thinking> tags before answering

</strategy>
"""



for _ in range(5):
    try: 
        base_poker_agent = ta.agents.AnthropicAgent(
            model_name="claude-3-7-sonnet-latest",
            system_prompt= "You are a professional poker player."
        )

        strategic_thought_poker_agent = ta.agents.wrappers.ThoughtAgentWrapper(
            agent=base_poker_agent,
            thought_prompt=poker_strategy,
            debugging=True
        )

        env = ta.make_online(
            env_id=["Poker-v0"], 
            model_name=model_name,
            model_description=model_description,
            email=email
        )
        env = ta.wrappers.LLMObservationWrapper(env=env)

        env.reset(num_players=1)

        done = False
        while not done:
            player_id, observation = env.get_observation()
            action = strategic_thought_poker_agent(observation)
            done, info = env.step(action=action)
        env.close()
        print(info)
        print(f"DONE - sleeping")
        time.sleep(10)
    except Exception as e:
        print(f"EXCEPTION: {e}")
        time.sleep(30)