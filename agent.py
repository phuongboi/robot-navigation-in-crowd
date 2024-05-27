import numpy as np

class Agent(object):
    def __init__(self):
        self.visible = None
        self.v_pref = None
        self.radius = None
        self.px = None
        self.py =None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = None

    def observable_state(self):
        return np.array([self.px, self.py, self.vx, self.vy, self.radius])

    def full_state(self):
        return np.array([self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta])

    def step(self, action):
        self.px = self.px + action[0] * self.time_step
        self.py = self.py + action[1] * self.time_step
        self.vx = action[0]
        self.vy = action[1]

    def set(self, px, py, gx, gy, vx, vy, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    def get_position(self):
        return self.px, self.py

    def get_goal_position(self):
        return self.gx, self.gy


class Human(Agent):
    def __init__(self, time_step):
        super().__init__()
        self.visible = True
        self.radius = 0.3
        self.time_step = time_step
        self.v_pref = 1

class Robot(Agent):
    def __init__(self, time_step):
        super().__init__()
        self.visible = False
        self.radius = 0.3
        self.time_step = time_step
        self.v_pref = 1
        self.px = 0
        self.py = -4
        self.gx = 0
        self.gy = 4
        self.vx = 0
        self.vy = 0
        self.theta = 0
