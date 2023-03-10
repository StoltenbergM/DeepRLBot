from stable_baselines3 import PPO
import pathlib
from rlgym.utils.action_parsers.discrete_act import DiscreteAction


class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = { ## Man ska ændre nogle ting inden man loader det ind, det vises her:
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
            "devide": "cpu"
        }
        self.actor = PPO.load(str(_path) + "models/18_01_Braindead.zip", custom_objects=custom_objects)
        self.parser = DiscreteAction()

    def act(self, state):
        # Evaluate your model here
        action = self.actor.predict(state, deterministic=True) ## Måske skal deterministic af
        # (Den afgør om den vælger den bedste action hver gang, ikke alle bots er bedst sådan
        x = self.parser.parse_actions(action[0], state)
        return x[0]