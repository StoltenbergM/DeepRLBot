from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

import numpy as np

CAR_MAX_SPEED = 2300
GOAL_Y = 5120 ## For Orange
BALL_RADIUS = 93
PENALTY = 0.55 ## Makes sure it will not be a positive thing to be behind the ball, this is a constant penalty if behind


class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:

        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        if player.team_num == 1:
            self.last_vel = state.ball.linear_velocity

        return reward


class RecoveryReward(RewardFunction):
    '''
    This will only kick in when a player is further away from its goal than the ball
    At that point, player will be rewarded with its velocity towards its own goal
    '''
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == 0: ## The blue goal is on -x coordinate
            NEGATIVE_BLUE = -1
        else:
            NEGATIVE_BLUE = 1

        player_goal_pos = np.array([0, GOAL_Y * NEGATIVE_BLUE, 0])

        ## Getting distances to compare
        dist_car_to_goal = np.linalg.norm(player.car_data.position - player_goal_pos) ## Distance from car to goal
        dist_ball_to_goal = np.linalg.norm(state.ball.position - player_goal_pos) + BALL_RADIUS ## Ball to goal

        if dist_car_to_goal > dist_ball_to_goal:
            vel = player.car_data.linear_velocity ## An vector (array) of linear vel
            pos_diff = player_goal_pos - player.car_data.position ## array of pos diff
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff) ## Normalised
            norm_vel = vel / CAR_MAX_SPEED ## divided by max
            reward = float(np.dot(norm_pos_diff, norm_vel)) - PENALTY ## Calculate reward, add a penalty

            return reward
        else:
            return 0

class KickoffReward(RewardFunction):
    def __init__(self, boost_punish: bool = True):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    ## To ensure only reward to closest in 2s and 3s, closest_to_ball is calc
    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        ## Get current players distance to ball
        player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != p.car_id: ## Meaning a tm8, but not opponent or self
                dist = np.linalg.norm(p.car_data.position - state.ball.position)
                if dist < player_dist:
                    return False

        return True

    ## Giving 1 or -1 if kickoff player is closest or not
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0

        ## If kickoff (ball center), and closest (meaning the players going for kickoff)
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
            player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
            boost = player.boost_amount ## Add reward for saving boost
            exponential_effect = 2000 / (player_dist + 2000) ## Making the reward give more, closer to the centrum

            for p in state.players:
                ## If itself
                if player.car_id == p.car_id: continue
                ## If player are closest to the ball
                if player_dist < np.linalg.norm(p.car_data.position - state.ball.position):
                    reward = exponential_effect + boost * exponential_effect ** 2
                else:
                    reward = -exponential_effect + boost * exponential_effect ** 2

        return reward
