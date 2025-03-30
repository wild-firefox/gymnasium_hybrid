# gymnasium-hybrid

强化学习混合动作空间环境，基于Gymnasium 1.0.0。

## 环境

- Moving-v0 
- Sliding-v0
- HardMove-v0 

## 安装方法

```bash
cd path/to/gymnasium_hybrid
pip install -e .
```
or 
```bash
pip install git+https://github.com/wild-firefox/gymnasium_hybrid.git
```
## 示例用法

```python
import gymnasium as gym
import gymnasium_hybrid

env = gym.make('Sliding-v0')
state, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        state, info = env.reset()

env.close()
```

## 致谢

本项目改编自 https://github.com/thomashirtz/gym-hybrid  
以及 https://github.com/opendilab/DI-engine/tree/main/dizoo/gym_hybrid/envs/gym-hybrid
