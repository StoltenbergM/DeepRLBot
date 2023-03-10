from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

BALL_MAX_X = 2500 # Both directions
BALL_MAX_Y = 2500
BALL_MIN_Z = 600
BALL_MAX_Z = 1600

CAR_DIST_X_MAX = 200
CAR_DIST_Y_MIN = 400
CAR_DIST_Y_MAX = 800

EXTRA_ADV = 200

CAR_BOOST_MIN = 0.20
CAR_BOOST_MAX = 1

IN_AIR_FREQ = 2 # 1 out of X shot
IN_AIR_HEI_MIN = 80
IN_AIR_HEI_MAX = 350
IN_AIR_BALL_HEIGHT = 900 # +- 100

class CloseAerialState(StateSetter):

    def __init__(self, one_car_advantage=False, reset_to_max_boost=False, start_in_air=True):
        """
        :param one_car_advantage: Will spawn one car closer/or more boost
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or random
        :param start_in_air: Will start players in the air going against the ball
        """
        super().__init__()
        self.team_turn = 0 ## For advantage True
        self.one_car_advantage = one_car_advantage
        self.reset_to_max_boost = reset_to_max_boost
        self.start_in_air = start_in_air

    def reset(self, state_wrapper: StateWrapper):

        self._reset_cars_and_ball(state_wrapper, self.team_turn, self.reset_to_max_boost, self.one_car_advantage
                                  , self.start_in_air)

        # which team will recieve the next advantage
        self.team_turn = (self.team_turn + 1) % 2

    def _reset_cars_and_ball(self, state_wrapper: StateWrapper, team_turn, reset_to_max_boost, one_car_advantage, start_in_air):

        try:
            in_air_pick = random.randrange(1, IN_AIR_FREQ + 1)
        except:
            in_air_pick = 1 ## Will always be shadow

        ## For cars: extra pos from ball
        x_extra = rand.random() * (CAR_DIST_X_MAX * 2) - CAR_DIST_X_MAX
        y_extra = rand.random() * (CAR_DIST_Y_MAX - CAR_DIST_Y_MIN) + CAR_DIST_Y_MIN

        # Getting all the variables needed for if start_in_air:
        if in_air_pick == 1 and start_in_air:
            # Ball z pos:
            pos_z = rand.random() * 100 + IN_AIR_BALL_HEIGHT

            # Car: Setting the z pos for the car in the air between min and max
            car_z = 17 + rand.random() * (IN_AIR_HEI_MAX - IN_AIR_HEI_MIN) + IN_AIR_HEI_MIN
            y_extra = y_extra * 1.4 ## Greater distance from ball
            tilt_up = (0.2 * np.pi + 0.3 * rand.random()) # negative for orange
            vel_y = 500 # negative for orange
            vel_z = 500
            boost = 1
        else:
            # Ball z pos:
            pos_z = rand.random() * (BALL_MAX_Z - BALL_MIN_Z) + BALL_MIN_Z

            # Car:
            car_z = 17
            tilt_up = 0
            vel_y = 0
            vel_z = 0
            boost = rand.random() * (CAR_BOOST_MAX - CAR_BOOST_MIN) + CAR_BOOST_MIN

        ## Get random ball spawn
        pos_x = rand.random() * (BALL_MAX_X * 2) - BALL_MAX_X
        pos_y = rand.random() * (BALL_MAX_Y * 2) - BALL_MAX_Y

        state_wrapper.ball.set_pos(pos_x, pos_y, pos_z)

        ## Some random vel
        ran_vel = 400 * rand.random() - 200
        state_wrapper.ball.set_lin_vel(ran_vel, ran_vel, ran_vel + 100)

        # Placing the cars:
        first_set = False # for if 2s or 3s
        for car in state_wrapper.cars:

            if car.team_num == 0 and not first_set: ## First blue car
                if one_car_advantage:
                    if car.team_num == team_turn: ## Tilføjer EXTRA_ADV, på en side, hvis one_car_advantage
                        car.set_pos(pos_x - x_extra, pos_y - y_extra - EXTRA_ADV, car_z)
                    else:
                        car.set_pos(pos_x - x_extra, pos_y - y_extra, car_z)
                else: ## Hvis ikke one_car_advantage
                    car.set_pos(pos_x - x_extra, pos_y - y_extra, car_z)
                car.set_rot(tilt_up, 0.5 * np.pi, 0)
                car.set_lin_vel(0, vel_y, vel_z)
                first_set = True

            else: ## Orange car
                if one_car_advantage:
                    if car.team_num == team_turn:
                        car.set_pos(pos_x + x_extra, pos_y + y_extra + EXTRA_ADV, car_z)
                    else:
                        car.set_pos(pos_x + x_extra, pos_y + y_extra, car_z)
                else: ## Hvis ikke one_car_advantage
                    car.set_pos(pos_x + x_extra, pos_y + y_extra, car_z)
                car.set_rot(tilt_up, -0.5 * np.pi, 0)
                car.set_lin_vel(0, -vel_y, vel_z)

            if reset_to_max_boost:
                car.boost = 100
            else:
                car.boost = boost ## samme boost for begge biler på denne måde
