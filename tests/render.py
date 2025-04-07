import time
import gymnasium as gym
import gymnasium_hybrid

if __name__ == '__main__':
    env = gym.make('Sliding-v0', render_mode='human')
    env.reset()

    # done = False
    # while not done:
    #     _, _, done, _ = env.step(env.action_space.sample())
    #     env.render()
    #     time.sleep(0.1)

    done = False
    while not done:
        action = env.action_space.sample()
        print((action[1].dtype))
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f'State: {state} Reward: {reward} Done: {done}')
        env.render()
        time.sleep(0.1)

    time.sleep(1)
    env.close()
