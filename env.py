import rvo2
import numpy as np
from numpy.linalg import norm
from agent import Human, Robot

class CrowdEnv(object):
    def __init__(self):
        self.human_list = []
        self.circle_radius = 4
        self.square_width = 10
        self.discomfort_dist = 0.2
        self.sim_time = 0
        self.time_out_duration = 25
        self.speed_samples = 5
        self.rotation_samples = 16
        # for orca simulation
        self.safety_space = 0 # no safety space for human
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.time_step = 0.25
        self.radius = 0.3
        self.max_speed = 1
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
        self.vel_samples = self.action_space()

    def generate_human_postion(self, human_num, rule):
        if  rule == "square":
            self.human_list = self.square_rule(human_num)
        elif rule == "cross":
            self.humans_list = self.cross_rule(human_num)

    def circle_rule(self, human_num):
        while True:
            human = Human(self.v_pref, self.radius, self.time_step)
            angle = np.random.randm() * np.pi *2
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise

            collide = False
            for agent in [self.robot] + self.human_list:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm(px - agent.px, py - agent.py) < min_dist or \
                    norm(px - agent.gx, py - agent.gy) < min_dist:
                    collide = True
                    break
            if not collide:
                human.set(px, py, -px, -py, 0, 0, 0)
                self.human_list.append(human)

            if len(self.human_list) >= human_num:
                break
    def square_rule(self, human_hum):
        while True:
            human = Human(self.v_pref, self.radius, self.time_step)
            px = (np.random.random() - 0.5) * self.square_width
            py = (np.random.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.human_list:
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm(px - agent.px, py - agent.py) < min_dist or \
                    norm(px - agent.gx, py - agent.gy) < min_dist:
                    collide = True
                    break
            if not collide:
                human.set(px, py, -px, -py, 0, 0, 0)
                self.human_list.append(human)

            if len(self.human_list) >= human:
                break

    def action_space(self):
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * self.max_speed for i in range(self.speed_samples)]
        print("sample speed",speeds)
        rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        action_space = []
        for rotation, speed in itertools.product(rotations, speeds):
            action_space.append([speed * np.cos(rotation), speed * np.sin(rotation)])

        return action_space

    def reset(self, human_num, test_phase=False, counter):
        self.robot = Robot(self.time_step)
        if test_phase:
            np.random.sead(counter)
        self.generate_human_postion(human_num)
        obs = [self.robot.full_state] + [human.observable_state() for human in self.human_list]
        assert len(obs) == 6 #debug
        # orca simultion
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        for human in self.human_list:
            self.sim.addAgent((human.px, human.py), *params, human.radius + 0.01 + self.safety_space,
                              human.v_pref, (human.vx, human.py))
        self.sim_time = 0
        return obs

    def step(self, action):
        for i, human in enumerate(self.human_list):
            v_pref  = np.array((human.gx, human.gy)) - np.array((human.px, human.py))
            if norm(v_pref) > 1:
                v_pref /= norm(v_pref)
            self.sim.setAgentPrefVelocity(i, tuple(v_pref))
            human.v_pref = v_pref

        self.sim.doStep()
        for i, human in enumerate(self.human_list):
            human.set_position(self.sim.getAgentPosition(i))
            human.set_velocity(self.sim.getAgentVelocity(i))
        self.robot.step(action)
        self.sim_time += self.time_step
        # compute reward
        distance_list = []
        for human in self.human_list:
            distance_list.append(norm(human.get_position - self.robot.get_position) - 2 * self.radius)
        dmin = min(distance_list)
        reaching_goal = norm(self.robot.get_position - self.robot.get_goal_position) < self.robot.radius

        if self.sim_time >= self.time_out_duration:
            reward = 0
            done = True
            info = "timeout"
        elif d_min < 0:
            reward = -0.25
            done = True
            info = "collide"
        elif d_min < self.discomfort_dist:
            reward = -0.1 + 0.05*d_min
            done = False
            info = "close"
        elif reaching_goal:
            reward = 1
            done = True
            info = "goal"
        else:
            reward = 0
            done False
            info = "onway"

        obs = [self.robot.full_state] + [human.observable_state() for human in self.human_list]

        return obs, reward, done, info
    def convert_coord(self, obs):
        assert len(obs) == 6
        robot_state = torch.Tensor(obs[0])
        human_state = torch.Tensor(np.array(obs[1:]))
        assert human_state.shape[0] = 5
        assert human_state.shape[1] = 5
        dx = robot_state[5] - robot_state[0]
        dy = robot_state[6] - robot_state[1]
        dg = torch.norm(dx, dy).expand(5,1)
        rot = torch.atan2(dy, dx).expand(5,1)
        v_pref  = robot_state[7].expand(5,1)
        vx = (robot_state[2] * torch.cos(rot) + robot_state[3] * torch.sin(rot)).expand(5,1)
        vy = (robot_state[3] * torch.cos(rot) - robot_state[2] * torch.sin(rot)).expand(5,1)
        radius = robot_state[4].expand(5,1)
        #theta = torch.zeros_like(v_pref)
        vx_human = (human_state[:, 2] * torch.cos(rot) + human_state[:, 3] * torch.sin(rot))
        vy_human = (human_state[:, 3] * torch.cos(rot) - human_state[:, 2] * torch.sin(rot))
        px_human = (human_state[:, 0] - robot_state[0]) * torch.cos(rot) + (human_state[:, 1] - robot_state[1]) * torch.sin(rot)
        py_human = (human_state[:, 1] - robot_state[1]) * torch.cos(rot) - (human_state[:, 0] - robot_state[0]) * torch.sin(rot)
        radius_human = human_state[:, 5]
        radius_sum = radius + radius_human
        da = torch.norm(human_state[:, 0] - robot_state[0], human_state[:, 1] - robot_state[1])

        new_state = torch.cat(dg, rot, vx, vy, v_pref, radius, px_human, py_human, vx_human, vy_human, radius_sum, da, radius_sum, dim=1)
        assert new_human_state.shape[0] == 5
        assert new_human_state.shape[1] == 13
        return new_state








        for i in range(len(obs)):

    s, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)

        return state


    def render(self, obs):
        print("show result", len(obs))