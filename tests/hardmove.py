import time
import gymnasium as gym
import gymnasium_hybrid

if __name__ == '__main__':
    env = gym.make('HardMove-v0')
    env.reset()

    ACTION_SPACE = env.action_space[0].n
    PARAMETERS_SPACE = env.action_space[1].shape[0]
    OBSERVATION_SPACE = env.observation_space.shape[0]

    # done = False
    # while not done:
    #     state, reward, done, info = env.step(env.action_space.sample())
    #     print(f'State: {state} Reward: {reward} Done: {done}')
    #     time.sleep(0.1)
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f'State: {state} Reward: {reward} Done: {done}')
        time.sleep(0.1)
