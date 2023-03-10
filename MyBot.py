import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

## Sample Setters
from rlgym.utils.state_setters import DefaultState, RandomState
# from rlgym_tools.extra_state_setters.wall_state import WallPracticeState ## Det her jeg har lavet min egen i stedet for wall
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

## Logger
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogRewardCallback

## Downloader rewards til rewarding function
from rlgym.utils.reward_functions.common_rewards.misc_rewards import AlignBallGoal
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward # ud fra forskellige event (goal, demo etc)
from rlgym.utils.reward_functions.common_rewards.misc_rewards import VelocityReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward # ud fra hastighed fra bot imod ball
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward # ud fra hastighed fra ball til modstander mål
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import SaveBoostReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import TouchBallReward
from rlgym.utils.reward_functions import CombinedReward # Kan kombinere

## Egne state setters
from aerial_state import AerialKOState
from recovery_state import RecoveryState
from dribbling_state import DribblingState
from shooting_state import ShootingState
from closeaerial_state import CloseAerialState
from goalline_state import GoaliePracticeState

## Egne rewards
from my_rewards import TouchVelChange
from my_rewards import RecoveryReward
from my_rewards import KickoffReward

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action (hvis den er lav, vil den tage meget lang tid om at træne)
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5 (er med til at give gamma)

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
## Hvor mange steps vil jeg træne den i per gang
    agents_per_match = 2 # Fordi jeg træner en 1v1 bot, og har self_train on (spawn_opponent)
    num_instances = 7 # Hvor mange fx Rocket League jeg vil åbne på samme tid (så mange som pc'en kan håndtere) Fx 10-20
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps

    print(f"fps={fps}, gamma={gamma})")

## Game Speed: sv_soccar_gamespeed 1  or sv_soccar_gamespeed 100

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  # 3v3 to get as many agents going as possible, will make results more noisy
            tick_skip=frame_skip,
            reward_function=SB3CombinedLogReward(    # Combined reward system
                (
                    TouchVelChange(),
                    RecoveryReward(),
                    KickoffReward(),
                    VelocityPlayerToBallReward(), # Returns the scalar projection of the agent's velocity vector on to the ball's position vector.
                    VelocityReward(), # Fart reward, simpel
                    # FaceBallReward(), # Returns positive reward scaled by the angle between the nose of the agent's car and the ball.
                    SaveBoostReward(), # Each step the agent is rewarded with sqrt(player.boost_amount)
                    TouchBallReward(aerial_weight=4.2), # Returns positive reward every time the agent touches the ball with an optional scaling factor for how high the ball was in the air when touched.
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=7.0, # 5.0
                        save=30.0,
                        demo=20.0,
                        boost_pickup=10,
                   ),
                ),
                (2.5, 0.02, 0.18, 0.01, 0.0072, 0.004, 0.06, 0.12, 0.62)), # Vægtning af rewards, rewardene er ganget med disse vægte, så kan modeleres ud fra det
            spawn_opponents=True, # False will play against All-Star bot
            terminal_conditions=[TimeoutCondition(round(fps * 100)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=WeightedSampleSetter((ShootingState(), CloseAerialState(), DribblingState(),
                                               RecoveryState(), AerialKOState(), GoaliePracticeState(),
                                               DefaultState(), RandomState()), (3, 2, 3, 2, 1, 1, 2, 1)),
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)   # Start "num_instances" instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    try: # Dette gøres så den loader videre på modellen i stedet for at lave en ny
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="cuda"  # Need to set device again (if using a specific one) #auto
        )
        print("Loaded existing model")
    except:
        from torch.nn import Tanh
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])], # En måde hvor man splitter træningen i 2 lag, pi er
        )
        ## Selve modellen:
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,                  # PPO calls for multiple epochs
            policy_kwargs = policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,       # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="cuda"                # Uses GPU if available # auto
        )
        print("Loaded new model")

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = [CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model")\
        ,SB3CombinedLogRewardCallback(["TouchVelChange", "Recovery", "Kickoff", "Vel_P2B", "Velocity", "SaveBoost", "Touch", "Vel_B2G", "Event"], file_location="./combinedlogfiles")]

        #
        #model.learn(100_000, callback=callback) # Hvor lang tid den skal træne          25_000_000
        #
        ## Now, if one wants to load a trained model from a checkpoint, use this function
        ## This will contain all the attributes of the original model
        ## Any attribute can be overwritten by using the custom_objects parameter,
        ## which includes n_envs (number of agents), which has to be overwritten to use a different amount
        #model = PPO.load(
        #    "policy/rl_model_1000002_steps.zip",
        ## Use reset_num_timesteps=False to keep going with same logger/checkpoints

    while True:
        model.learn(10_000_000, callback=callback, reset_num_timesteps=False) # , reset_num_timesteps=False       100_000_000
        model.save("models/exit_save")
        model.save(f"mmr_models/{model.num_timesteps}")


'''
TO DO:
Gennemgå statesetters, se om de skal adjustes.
Mere variation på shooting og dribbling? Flere skud muligheder fra forskellige pos
Reward for flip? 
Tjek om recovery virker som det skal
Tjek om closeaerial er god nok
Setter med mere normale/common situationer, bold på jorden, få et angreb mere spændende
Øve at vende, powerslide eller halfflip, setter med ryggen til bold, og tvunget til at halfflip
Flip mulighed detector, lav fart, langt fra bold, rigtig retning, point for flip

Lots of handy scripts:
https://github.com/RLGym/rlgym-tools

https://rlgym.org/docs-page.html#reward-functions

https://github.com/RLBot/RLBot/wiki/Useful-Game-Values

https://github.com/Impossibum/kaiyo-bot

700.000 steps var cirka 80 min
Virker til at der er cirka 8750 steps i minuttet
1.000.000 steps cirka 114 min
1.000.000.000 ticks = 66.666.667 sec = 1.111.111 min = 18.518 hours = 772 days

5.000.000 tager ca 50 min med 8 instances in-real time, svarende til 5.555 min in-game time, dvs:
in-game time er 111 gange hurtigere end in-real time, svarende til at 1B steps skal køre i 167 timer irt

DONE:
Personlig kickoff, få bevægelse på bolden udover højde
Tilføj reward for ændring af fart på bolden, powershots
TouchVelChange up, touch ned
Lav en state til dribbling (flick)
Lave en reward der belønner at recover hele vejen hjem (rotere)
'''

