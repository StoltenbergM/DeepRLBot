from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

X_MAX = 1200
Y_MAX = 1200
Z_MAX_BALL = 150
MIN_HEIGHT = 800
CAR_BOOST_MIN = 0

class ShootingState(StateSetter):

    SPAWN_BLUE_POS = [[-2000, 1000, 17], [2000, 1000, 17], [0, -3000, 17], [0, 0, 17], ## 4 first positions for attack
                      [-1000, -4700, 17], [1000, -4700, 17], [0, -3700, 17], [0, -4200, 17]] ## 4 last is for defense
    SPAWN_BLUE_YAW = [0.4 * np.pi, 0.6 * np.pi, 0.5 * np.pi, 0.5 * np.pi,
                      0.3 * np.pi, 0.7 * np.pi, (rand.random() * 0.5 + 0.25) * np.pi, (rand.random() * 0.5 + 0.25) * np.pi]
    SPAWN_ORANGE_POS = [[-1000, 4700, 17], [1000, 4700, 17], [0, 3700, 17], [0, 4200, 17], ## 4 first for defense
                        [-2000, -1000, 17], [2000, -2000, 17], [0, 3000, 17], [0, 0, 17]] ## 4 last for attack
    SPAWN_ORANGE_YAW = [-0.4 * np.pi, -0.6 * np.pi, -0.5 * np.pi, -0.5 * np.pi,
                      -0.3 * np.pi, -0.7 * np.pi, (rand.random() * 0.5 - 0.75) * np.pi, (rand.random() * 0.5 - 0.75) * np.pi]

    SPAWN_BALL = [[-2000, 2000, 17], [2000, 2000, 17], [0, -2000, 17], [0, 1000, 17], ## Ball spawning in front of attacker
                      [-2000, -2000, 17], [2000, -2000, 17], [0, 2000, 17], [0, -1000, 17]]
    RANDOM_SIDE = [1, -1]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4, 5, 6, 7]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]] + (rand.random() * 0.2 - 0.1) * np.pi
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]] + (rand.random() * 0.2 - 0.1) * np.pi
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = rand.random() * (1 - CAR_BOOST_MIN) + CAR_BOOST_MIN # Mere random boost

        self.reset_ball(state_wrapper, spawn_inds)

    def reset_ball(self, state_wrapper: StateWrapper, spawn_inds):
        """
        Function to set the ball around the attacker.

        :param state_wrapper: StateWrapper object to be modified.
        :param spawn_inds: index of the shot-selection list.
        """
        ball_count = 0

        random_side = random.choice(self.RANDOM_SIDE) ## To make sure the shots comes from both sides

        pos = self.SPAWN_BALL[spawn_inds[ball_count]]
        ball_count += 1

        pos_x = rand.random() * 300 - 150 + pos[0] + 1500 * random_side ## The x-value of the ball
        pos_y = rand.random() * 400 - 200 + pos[1]
        pos_z = rand.random() * 900

        state_wrapper.ball.set_pos(pos_x, pos_y, pos_z)
        state_wrapper.ball.set_lin_vel(rand.random() * 600 - 300 - 1100 * random_side, rand.random() * 200 - 100, rand.random() * 500 - 250) ## fart på bolden
        state_wrapper.ball.set_ang_vel(*rand_vec3(6)) ## spin på bolden