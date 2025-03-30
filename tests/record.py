import gymnasium as gym
import gym_hybrid

if __name__ == '__main__':
    env = gym.make('Sliding-v0', render_mode="rgb_array")
    # 使用RecordVideo替代Monitor
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder="./video",
        episode_trigger=lambda episode_id: True  # 录制每一个episode
    )
    
    # 不需要手动设置metadata，应在环境类中设置
    # env.metadata["render.modes"] = ["human", "rgb_array"]
    
    env.reset()

    terminated = truncated = False
    while not (terminated or truncated):
        # gymnasium的step返回5个值
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())

    env.close()

