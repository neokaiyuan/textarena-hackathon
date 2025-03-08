import asyncio
import textarena as ta
from mcp_agent import MCPAgent
from dotenv import load_dotenv

load_dotenv()


model_name = "Just For Fun"
model_description = "destroy"
email = "jednghk@gmail.com"

SYSTEM_PROMPT = "You are a professional player in the following games: Nim, Poker, Simple Negotiation, Spelling bee, Snake, Truth and Deception"

SPELLING_BEE_PROMPT = """
<spelling_bee_strategy>
You are a strategic Spelling Bee game competitor. This is a turn-based word game where players create valid English words using only allowed letters, with each new word needing to be at least as long as the previous one.

STRATEGIC FRAMEWORK:
1. EARLY GAME:
   • Start with minimal-length valid words (typically 3-4 letters)
   • Avoid words with common suffixes (-ing, -ed, -s) that opponents can easily extend
   • Strategically use uncommon letters early to preserve common ones (a, e, i, o, r, s, t) for later moves

2. MID-GAME:
   • Force incremental length increases (+1 letter when possible)
   • Track word length progression and plan 2-3 moves ahead
   • Identify potential endgame words and protect your path to them

3. ENDGAME:
   • Reserve longest possible words for final moves
   • Target words with difficult-to-extend letter combinations
   • Look for opportunities to create "dead ends" with unusual letter patterns

DECISION PROCESS (Apply for each turn):
• Analyze the current allowed letter set thoroughly
• Consider the minimum required word length (must match or exceed previous word)
• Evaluate your options based on:
  - Words that force difficult length jumps for opponent
  - Words that use strategic letter combinations
  - Words that limit opponent's future options

ALWAYS include your strategic reasoning before submitting your move, explaining:
1. Why you selected this specific word
2. What strategic advantage it provides
3. How it positions you for future moves

Format your final word selection in square brackets: [word]

Remember: Read game instructions carefully and follow the required format exactly.
</spelling_bee_strategy>
"""

SIMPLE_NEGO_PROMPT = """
<simple_negotiation_game_strategy>
You are Warren Buffet. Make sure you read the game instructions carefully, and always follow the required format.
</simple_negotiation_game_strategy>
"""

POKER_PROMPT = """
<poker_strategy>
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
ta
</poker_strategy>
"""

# Initialize agents
agent = MCPAgent(model_name="claude-3-7-sonnet-latest", system_prompt='\n'.join([POKER_PROMPT, SIMPLE_NEGO_PROMPT, SPELLING_BEE_PROMPT]))

# thought_agent_wrapper = ta.agents.wrappers.ThoughtAgentWrapper(
#     agent=agent,
#     thought_prompt='\n'.join([POKER_PROMPT, SIMPLE_NEGO_PROMPT]),
#     debugging=True
# )


# Initialize environment from subset and wrap it
env = ta.make_online(
    env_id=[
        "SpellingBee-v0",
        "SimpleNegotiation-v0",
        "Poker-v0",
    ],
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
