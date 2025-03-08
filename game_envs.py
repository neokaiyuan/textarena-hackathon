from enum import Enum, auto

class GameEnv(Enum):
    """Enum representing available game environments."""
    SPELLING_BEE = "SpellingBee-v0"
    SIMPLE_NEGOTIATION = "SimpleNegotiation-v0"
    POKER = "Poker-v0"