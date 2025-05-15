from gymnasium.envs.registration import register # 将gym 改为gymnasium
from gymnasium_hybrid.environments import MovingEnv
from gymnasium_hybrid.environments import SlidingEnv
from gymnasium_hybrid.environments import HardMoveEnv

# register(
#     id='Moving-v0',
#     entry_point='gym_hybrid:MovingEnv',
# )
register(
    id='Moving-v0',
    entry_point='gymnasium_hybrid:MovingEnv',
)

register(
    id='Sliding-v0',
    entry_point='gymnasium_hybrid:SlidingEnv',
)
register(
    id='HardMove-v0',
    entry_point='gymnasium_hybrid:HardMoveEnv',
)
