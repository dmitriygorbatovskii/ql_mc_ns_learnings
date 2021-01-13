import lgsvl
import os
from math import sqrt


class LgsvlEnv():

    def __init__(self):
        self.env = lgsvl.Simulator(os.environ.get("SIMULATOR_HOST", "127.0.0.1"), 8181)
        # загрузка сцены
        if self.env.current_scene == "BorregasAve":
            self.env.reset()
        else:
            self.env.load("BorregasAve")
        self.control = lgsvl.VehicleControl()
        self.vehicles = dict()

    def reset(self):
        self.done = False
        self.steps = 0
        self.env.reset()
        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(0, -2, 0)
        state.transform.rotation.x = 0
        state.transform.rotation.y = 180
        state.transform.rotation.z = 0
        self.ego = self.env.add_agent(name="Lincoln2017MKZ (Apollo 5.0)", agent_type=lgsvl.AgentType.EGO, state = state)
        self.px = self.ego.transform.position.x
        self.py = self.ego.transform.position.y
        self.pz = self.ego.transform.position.z

        state.transform.position = lgsvl.Vector(15, -2, -15)
        npc = self.env.add_agent("Sedan", lgsvl.AgentType.NPC, state = state)
        return self.get_observation()

    def get_observation(self):
        self.x = self.ego.transform.position.x
        self.y = self.ego.transform.position.y
        self.z = self.ego.transform.position.z
        return [round(self.x, 1), round(self.y, 1), round(self.z, 1)]

    def step(self, action):
        self.info = {}
        self.reward = 0

        if action == 0:
            self.control.steering += 0.0  # поворот
            self.control.throttle += 0.0  # заслонка
            self.control.braking += 0.0  # тормоза
            self.reward -= 1
        elif action == 1 and self.control.steering < 1:
            self.control.steering += 0.1
            #self.control.braking += 0.1
        elif action == 2 and self.control.steering > -1:
            self.control.steering -= 0.1
            #self.control.braking += 0.1
        elif action == 3 and self.control.throttle < 1:
            self.control.braking = 0.0
            self.control.throttle += 0.1
        elif action == 4 and self.control.braking > 1:
            self.control.throttle = 0.0
            self.control.braking += 0.1
        self.ego.apply_control(self.control, sticky=True)

        if self.control.throttle <= 0:
            self.reward -= 1

        self.env.run(0.1)
        self.steps += 1
        if self.steps == 200:
            self.reward -= 10
            self.done = True
        self.calculate_reward()

        #if self.x in [-1, 0, 1] and self.y in [-4, -3, -2] and self.z in [-31, -30, -29]:
        if sqrt((self.x - 15)**2 + (self.y + 2)**2 + (self.z + 15)**2) < 9:
            self.reward += 50
            print(1)
            self.done = True

        self.px = self.x
        self.py = self.y
        self.pz = self.z

        return self.get_observation(), self.reward, self.done, self.info

    def calculate_reward(self):
        a = sqrt((self.px - 15)**2 + (self.py + 2)**2 + (self.pz + 15)**2)
        b = sqrt((self.x - 15)**2 + (self.y + 2)**2 + (self.z + 15)**2)
        if a == b:
            self.reward -= 1
        else:
            self.reward += (a-b)*5

    def _on_collision(self, agent1, agent2, contact):
        self.reward -= 50
        self.done = True
        name1 = self.vehicles[agent1]
        name2 = self.vehicles[agent2] if agent2 is not None else "OBSTACLE"
        #print("{} collided with {} at {}".format(name1, name2, contact))




