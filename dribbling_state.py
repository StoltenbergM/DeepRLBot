from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.math import rand_vec3
import numpy as np
from numpy import random as rand
import random

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 3000 ## Maximum half of this
PLACEMENT_BOX_Y_OFFSET = 3000

YAW_MIN = 0.30 * np.pi
YAW_MAX = 0.70 * np.pi

PLACEMENT_DEFENDER_BOX_X = 4000
PLACEMENT_DEFENDER_BOX_Y = 5000
PLACEMENT_DEFENDER_BOX_Y_OFFSET = -2000

CAR_BOOST_MAX = 0.80
CAR_BOOST_MIN = 0.05

OCTANE_HEIGHT = 36
BALL_RADIUS = 95 ## 92.75 eller 94.41

SHADOW_FREQUENCY = 4
SH_Y_MIN = 600 # Minimum distance of the bouncedribble at y
SH_Y_MAX = 2000

BOUNCE_BALL_FREQUENCY = 4 ## 1 out of X shot is bounce, if bounce_dribble_allowed is True
BO_Y_MIN = 400 # Minimum distance of the bouncedribble at y
BO_Y_MAX = 800
BO_Z_MIN = 300 ## hight of it
BO_Z_MAX = 600
BOUNCE_VEL = 400 ## Max 50%

class DribblingState(StateSetter):

    def __init__(self, reset_to_max_boost=False, bounce_dribble_allowed=True, shadow_defense=True):
        """
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        :param bounce_dribble_allowed: Will bounce the ball in front of attacker at 20% of the shot at default
        :param shadow_defense: Will put the defender in shadow position
        """
        super().__init__()
        self.team_turn = 0  # swap every reset who's getting shot at

        self.reset_to_max_boost = reset_to_max_boost
        self.bounce_dribble_allowed = bounce_dribble_allowed
        self.shadow_defense = shadow_defense

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new shot

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_cars_and_ball(state_wrapper, self.team_turn, self.reset_to_max_boost, self.bounce_dribble_allowed
                                  , self.shadow_defense)

        # which team will recieve the next incoming dribble
        self.team_turn = (self.team_turn + 1) % 2

    def _reset_cars_and_ball(self, state_wrapper: StateWrapper, team_turn, reset_to_max_boost, bounce_dribble_allowed, shadow_defense):
        """
        Function to set cars in preparation for an incoming shot

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        ## The blue goal is on -y coordinate
        NEGATIVE_IF_ORANGE = (1 if team_turn == 0 else -1)

        first_set = False ## For more than 1v1s

        ## For shadow defense - picking the frequency, 1 out of X shots
        try:
            shadow_shot_pick = random.randrange(1, SHADOW_FREQUENCY + 1)
        except:
            shadow_shot_pick = 1 ## Will always be shadow

        # Is outside of the loop, as it can be used for shadow also:
        x_pos, y_pos = self._get_xy_for_attacker(team_turn)  ## Finder x og y for både bil og bold
        yaw = (rand.random() * (YAW_MAX - YAW_MIN) + YAW_MIN) * NEGATIVE_IF_ORANGE ## retning af bil

        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set: ## Attacker (they take turns)
                first_set = True
                car.set_pos(x_pos, y_pos, z=17)
                car.set_rot(0, yaw, 0) ## Rotation mellem yaw min og max

            else: ## Defender
                ## If shadow defending:
                if shadow_shot_pick == 1 and shadow_defense:
                    ## Making the defender always be on the inside, if shadow on:
                    x_pos_def = (x_pos + 1200 * rand.random() if x_pos < 0 else x_pos - 1200 * rand.random())
                    y_pos_def = y_pos + ((SH_Y_MAX - SH_Y_MIN) * rand.random() + SH_Y_MIN) * NEGATIVE_IF_ORANGE
                    car.set_rot(0, yaw, 0)
                else:
                    x_pos_def, y_pos_def = self._get_xy_for_defender(team_turn)
                    car.set_rot(0, rand.random() * np.pi, 0)  ## Rotation mellem yaw min og max
                car.set_pos(x_pos_def, y_pos_def, z=17)


            if reset_to_max_boost:
                car.boost = 100
            else:
                car.boost = rand.random() * (CAR_BOOST_MAX - CAR_BOOST_MIN) + CAR_BOOST_MIN ## samme for boost

        ## Picking state from frequency
        # This is only for if bounce_dribble is on, and the frequency will be decided here, 1 out of X
        try:
            bouncepick = random.randrange(1, BOUNCE_BALL_FREQUENCY + 1)
        except:
            bouncepick = 1 ## randrange cant take range of 1, 1

        # If Bounce dribble is on, the ball will spawn at a greater distance to the attacking car:
        # The coordinates of the ball is decided:
        if bouncepick == 1 and bounce_dribble_allowed == True: ## Adding extra distances for bounce_dribbles
            try:
                x_pos_ball = x_pos + rand.random() * 200 - 100 ## From 0 - 200 difference random
                y_pos_ball = y_pos + (rand.random() * (BO_Y_MAX - BO_Y_MIN) + BO_Y_MIN) * NEGATIVE_IF_ORANGE
                z_pos_ball = OCTANE_HEIGHT + BALL_RADIUS + rand.random() * (BO_Z_MAX - BO_Z_MIN) + BO_Z_MIN
            except Exception as e:
                print(e)
        else: ## without bounce_dribble
            x_pos_ball = x_pos
            y_pos_ball = y_pos + rand.random() * 50 * NEGATIVE_IF_ORANGE
            z_pos_ball = OCTANE_HEIGHT + BALL_RADIUS + rand.random() * 60

        ## Spawning the ball
        pos = np.array([x_pos_ball, y_pos_ball, z_pos_ball])
        if bouncepick == 1 and bounce_dribble_allowed == True: ## Add a little power on the ball on bounces
            random_pace = rand.random() * BOUNCE_VEL - (BOUNCE_VEL / 2) ## Random pace, max 50% of BOUNCE_VEL_MAX
            lin_vel = np.array([random_pace, random_pace, random_pace])
        else:
            lin_vel = np.array([0, 0, 0])
        ang_vel = np.array([0, 0, 0])

        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

    def _get_xy_for_attacker(self, team_delin):
        """
        Function to place a car in an allowed areaI

        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        x_pos = rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X / 2
        y_pos = rand.random() * PLACEMENT_BOX_Y - PLACEMENT_BOX_Y / 2

        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET

        return x_pos, y_pos

    def _get_xy_for_defender(self, team_delin):
        """
        Function to place a car in an allowed areaI

        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        x_pos_def = rand.random() * PLACEMENT_DEFENDER_BOX_X - PLACEMENT_DEFENDER_BOX_X / 2
        y_pos_def = rand.random() * PLACEMENT_DEFENDER_BOX_Y - PLACEMENT_DEFENDER_BOX_Y / 2

        if team_delin == 0:
            y_pos_def -= PLACEMENT_DEFENDER_BOX_Y_OFFSET
        else:
            y_pos_def += PLACEMENT_DEFENDER_BOX_Y_OFFSET

        return x_pos_def, y_pos_def
