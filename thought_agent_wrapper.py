import asyncio
import re 
from typing import Optional, Any, List, Tuple 

import textarena as ta 

class ThoughtAgentWrapper(ta.AgentWrapper):
    """ TODO """
    def __init__(
        self, 
        agent:ta.Agent, 
        thought_prompt: Optional[str] = None, 
        answer_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        """ TODO """
        super().__init__(agent)

        self.agent_system_prompt = self.agent.system_prompt
        self.thought_prompt = thought_prompt if thought_prompt is not None else (
            "\nPlease think extensively about what you want to do next. Analyze your current position, "
            "you strategy, what your opponents strategy might be and what you should do next to maximize "
            "your chance of winning."
        )

        self.answer_prompt = answer_prompt if answer_prompt is not None else (
            "\nGiven the game observations, and your above thoughts, please give the reply you want "
            "to submit to the game. Make sure you follow all rules and necessary formats."
        )

        self.debugging = debugging 


    async def __call__(self, observation: str) -> str:
        """ TODO """

        # set agent prompt 
        self.agent.system_prompt = self.thought_prompt

        # first forward
        thoughts = asyncio.get_event_loop().run_until_complete(await self.agent(observation + f"\n\nThoughts: "))
        if self.debugging:
            print(f"\n\nAgent thoughts: {thoughts}")


        # set agent prompt 
        self.agent.system_prompt = self.answer_prompt 

        # second forward
        answer = asyncio.get_event_loop().run_until_complete(await self.agent(observation + f"\n\nThoughts: {thoughts}" + self.answer_prompt))
        if self.debugging:
            print(f"\n\nAnswer: {answer}")
        return answer 

