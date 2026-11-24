"""Policy and value networks."""

from .human import HumanActor, HumanCritic
from .zombie import ZombieActor, ZombieCritic

__all__ = ["HumanActor", "HumanCritic", "ZombieActor", "ZombieCritic"]
