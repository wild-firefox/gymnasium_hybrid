from collections import namedtuple
from typing import Optional
from typing import Tuple

import gymnasium as gym #gym
import numpy as np
import cv2
import os
from gymnasium import spaces
from gymnasium.utils import seeding

# gym.logger.set_level(40)  # noqa

from .agents import BaseAgent, MovingAgent, SlidingAgent, HardMoveAgent

# Action Id
ACCELERATE = 0
TURN = 1
BREAK = 2

Target = namedtuple('Target', ['x', 'y', 'radius'])


class Action:
    """"
    Action class to store and standardize the action for the environment.
    """

    def __init__(self, id_: int, parameters: list):
        """"
        Initialization of an action.

        Args:
            id_: The id of the selected action.
            parameters: The parameters of an action.
        """
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        """"
        Property method to return the parameter related to the action selected.

        Returns:
            The parameter related to this action_id
        """
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class BaseEnv(gym.Env):
    """"
    Gym environment parent class.
    """
    # 添加这个metadata声明
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        seed: Optional[int] = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1,
        render_mode: Optional[str] = None,  # 添加这一行
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = BaseAgent(break_value=break_value, delta_t=delta_t)

        parameters_min = np.array([0, -1], dtype=np.float32)
        parameters_max = np.array([1, +1], dtype=np.float32)

        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(parameters_min, parameters_max,dtype=np.float32)))
        self.observation_space = spaces.Box(-np.ones(10), np.ones(10), dtype=np.float32)
        dirname = os.path.dirname(__file__)
        self.bg = cv2.imread(os.path.join(dirname, 'bg.jpg'))
        self.bg = cv2.cvtColor(self.bg, cv2.COLOR_BGR2RGB)
        self.bg = cv2.resize(self.bg, (800, 800))
        self.target_img = cv2.imread(os.path.join(dirname, 'target.png'), cv2.IMREAD_UNCHANGED)
        self.target_img = cv2.resize(self.target_img, (60, 60))

        self.render_mode = render_mode  # 保存渲染模式
        self.window = None
        self.clock = None


    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0

        limit = self.field_size - self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return np.array(self.get_state(), dtype=np.float32) ,{}

    def step(self, raw_action: Tuple[int, list]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), 0)
            self.agent.accelerate(acceleration)
        elif action.id == BREAK:
            self.agent.break_()

        if self.distance < self.target_radius and self.agent.speed == 0:
            reward = self.get_reward(last_distance, True)
            terminated = True #done = True
            truncated = False
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y
                                                        ) > self.field_size or self.current_step > self.max_step:
            reward = -1
            terminated = False
            truncated = True# done = True
        else:
            reward = self.get_reward(last_distance)
            terminated = False
            truncated = False

        return np.array(self.get_state(), dtype=np.float32), reward, terminated, truncated, {}

    def get_state(self) -> list:
        state = [
            self.agent.x, self.agent.y, self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta), self.target.x, self.target.y, self.distance,
            0 if self.distance > self.target_radius else 1, self.current_step / self.max_step
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)).item()

        # 修改render方法
    def render(self):
        """根据render_mode渲染环境"""
        if self.render_mode is None:
            return
        
        screen_width = 400
        screen_height = 400
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        agent_radius = 0.05

        try:
            import pygame
            from pygame import gfxdraw
            
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode((screen_width, screen_height))
                if hasattr(self, "clock"):
                    self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface((screen_width, screen_height))
            canvas.fill((255, 255, 255))  # 填充白色背景
            
            # 画目标点
            target_x = int(unit_x * (1 + self.target.x))
            target_y = int(unit_y * (1 + self.target.y))
            target_radius = int(unit_x * self.target_radius)
            gfxdraw.aacircle(
                canvas, target_x, target_y, target_radius, (0, 153, 0)
            )
            
            # 画智能体
            agent_x = int(unit_x * (1 + self.agent.x))
            agent_y = int(unit_y * (1 + self.agent.y))
            agent_radius_px = int(unit_x * agent_radius)
            gfxdraw.filled_circle(
                canvas, agent_x, agent_y, agent_radius_px, (25, 76, 229)
            )
            
            # 画方向箭头
            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            theta = self.agent.theta
            arrow_x = agent_x + int(t * np.cos(theta))
            arrow_y = agent_y + int(t * np.sin(theta))
            pygame.draw.line(
                canvas, 
                (0, 0, 0), 
                (agent_x, agent_y), 
                (arrow_x, arrow_y), 
                2
            )
            
            # 添加背景和边框
            if hasattr(self, "bg") and self.bg is not None:
                bg_surface = pygame.surfarray.make_surface(
                    np.transpose(
                        cv2.resize(self.bg, (screen_width, screen_height)), 
                        (1, 0, 2)
                    )
                )
                # 只替换白色区域
                canvas_array = pygame.surfarray.array3d(canvas)
                white_mask = (canvas_array == np.array([255, 255, 255])).all(axis=2)
                if white_mask.any():
                    bg_array = pygame.surfarray.array3d(bg_surface)
                    canvas_array[white_mask] = bg_array[white_mask]
                    new_surface = pygame.surfarray.make_surface(canvas_array)
                    canvas = new_surface
            
            # 添加边框
            pygame.draw.rect(canvas, (60, 60, 30), pygame.Rect(0, 0, screen_width, 6))
            pygame.draw.rect(canvas, (60, 60, 30), pygame.Rect(0, 0, 6, screen_height))
            pygame.draw.rect(canvas, (60, 60, 30), pygame.Rect(0, screen_height-6, screen_width, 6))
            pygame.draw.rect(canvas, (60, 60, 30), pygame.Rect(screen_width-6, 0, 6, screen_height))
            
            # 显示到窗口
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            
            # 根据渲染模式返回不同结果
            if self.render_mode == "rgb_array":
                return np.transpose(
                    pygame.surfarray.array3d(self.window), axes=(1, 0, 2)
                )
            elif self.render_mode == "human":
                # human模式下控制帧率
                if self.clock is not None:
                    self.clock.tick(self.metadata["render_fps"])
                return None
                
        except ImportError:
            # 如果无法导入pygame，返回None
            return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class MovingEnv(BaseEnv):

    def __init__(
        self,
        seed: int = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1,
        render_mode: Optional[str] = None,  # 添加这一行
    ):
        super(MovingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            render_mode=render_mode,  # 添加这一行
        )

        self.agent = MovingAgent(
            break_value=break_value,
            delta_t=delta_t,
        )


class SlidingEnv(BaseEnv):

    def __init__(
        self,
        seed: int = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 200,
        penalty: float = 0.001,
        break_value: float = 0.1,
        render_mode: Optional[str] = None,  # 添加这一行
    ):
        super(SlidingEnv, self).__init__(
            seed=seed,
            max_turn=max_turn,
            max_acceleration=max_acceleration,
            delta_t=delta_t,
            max_step=max_step,
            penalty=penalty,
            break_value=break_value,
            render_mode=render_mode,  # 添加这一行
        )

        self.agent = SlidingAgent(break_value=break_value, delta_t=delta_t)


class HardMoveEnv(gym.Env):
    """"
    HardMove environment. Please refer to https://arxiv.org/abs/2109.05490 for details.
    """
    # 添加metadata
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_actuators: int = 4,
        seed: Optional[int] = None,
        max_turn: float = np.pi / 2,
        max_acceleration: float = 0.5,
        delta_t: float = 0.005,
        max_step: int = 25,
        penalty: float = 0.001,
        break_value: float = 0.1,
    ):
        """Initialization of the gym environment.

        Args:
            seed (int): Seed used to get reproducible results.
            max_turn (float): Maximum turn during one step (in radian).
            max_acceleration (float): Maximum acceleration during one step.
            delta_t (float): Time duration of one step.
            max_step (int): Maximum number of steps in one episode.
            penalty (float): Score penalty given at the agent every step.
            break_value (float): Break value when performing break action.
        """
        # Agent Parameters
        self.num_actuators = num_actuators
        self.max_turn = max_turn
        self.max_acceleration = max_acceleration
        self.break_value = break_value

        # Environment Parameters
        self.delta_t = delta_t
        self.max_step = max_step
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = penalty

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = HardMoveAgent(break_value=break_value, delta_t=delta_t, num_actuators=self.num_actuators)

        parameters_min = np.array([-1 for i in range(self.num_actuators)], dtype=np.float32)
        parameters_max = np.array([+1 for i in range(self.num_actuators)], dtype=np.float32)

        self.action_space = spaces.Tuple(
            (spaces.Discrete(int(2 ** self.num_actuators)), spaces.Box(parameters_min, parameters_max,dtype=np.float32))  
        )
        self.observation_space = spaces.Box(-np.ones(10), np.ones(10), dtype=np.float32)

    def seed(self, seed: Optional[int] = None) -> list:
        self.np_random, seed = seeding.np_random(seed)  # noqa
        return [seed]

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0

        limit = self.field_size - self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return np.array(self.get_state(),dtype=np.float32), {}

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, bool, dict]:
        move_direction_meta = raw_action[0]  # shape (1,) in {2**n}
        move_distances = raw_action[1]  # shape (2**n,)
        last_distance = self.distance
        self.current_step += 1

        self.agent.move(move_direction_meta, move_distances)
        if self.distance < self.target_radius:
            reward = self.get_reward(last_distance, True)
            terminated = True #done = True
            truncated = False
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y
                                                        ) > self.field_size or self.current_step > self.max_step:
            reward = -1
            terminated = False #done = True
            truncated = True
        else:
            reward = self.get_reward(last_distance)
            terminated = False 
            truncated = False

        return np.array(self.get_state(), dtype=np.float32), reward, terminated, truncated, {}
    def get_state(self) -> list:
        state = [
            self.agent.x, self.agent.y, self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta), self.target.x, self.target.y, self.distance,
            0 if self.distance > self.target_radius else 1, self.current_step / self.max_step
        ]
        return state

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)).item()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
