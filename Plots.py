import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import ast

Timesteps = []

Number_Reward_Func = 9
weights = [2.5, 0.02, 0.18, 0.01, 0.0072, 0.004, 0.06, 0.12, 0.62]
titles = ["TouchVelChange", "Recovery", "Kickoff", "Velolicty_P2B", "Velocity", "SaveBoost", "Touch", "Velocity_B2G", "Event"]
avg = 3000 ## Running average size

rewards = []
for reward_fun in titles:
    rewards.append([])
sum_rewards = []

f = open("./combinedlogfiles/rewards.txt", "r") ## alle rewards

## Creating a list of the lines in the reward text file
for i, row in enumerate(f):
    row = row.strip('][').split(', ') ## Making the string into a list
    ## Apparently, does the last item in the list not delete the ]

    for nu, number in enumerate(row): # This whole thing is done to get the ] deleted in the last object
        try:
            float(number)
        except:
            number = number[0:-2] # Deleting the last  ]
            row[nu] = number # Replacing the number

    if not type(row) is list: ## To make sure no errors
        continue
    if not len(row) is Number_Reward_Func: ## Makes sure no errors, doesnt include lines with not correct len
        continue

    Timesteps.append(i*600)

    for i, n in enumerate(row):
        rewards[i].append(float(row[i]))


x = Timesteps

## Add lists together to get sum
sum_rewards = [sum(x) for x in zip(*rewards)]

colors = ["blue","darkgreen","darkorange","lawngreen","firebrick","magenta","grey","yellow", "brown"]

plt.title("All Rewards")
plt.xlabel("time")
plt.ylabel("reward")
for i, reward_function in enumerate(range(Number_Reward_Func)):
    rewards[i] = pd.DataFrame(rewards[i]) # Making the list a pandas dataframe
    rewards[i] = rewards[i].rolling(avg, center=True).mean() # Meaning, rolling average
    plt.plot(x, rewards[i], colors[i], label = f'{titles[i]} - Weight: {weights[i]}')
plt.legend()
plt.grid(which="both")

# plot the sum:
sum_rewards = pd.DataFrame(sum_rewards)
sum_rewards = sum_rewards.rolling(avg, center=True).mean()

# On seperate y-axis
ax2 = plt.twinx()
ax2.plot(x, sum_rewards, "black")

plt.show()

'''
17166 lines in rewards1
weights = [0.2, 0.01, 0.6, 1, 1]
titles = ["Velocity_P2B", "Velocity_P", "Touch", "Velocity_B2G", "Event"]

10982 lines in rewards2
weights = [0.02, 0.01, 1, 1, 1]
titles = ["Velocity_P2B", "Velocity_P", "Touch", "Velocity_B2G", "Event"]

14-01-2023
VelocityPlayerToBallReward(),VelocityReward(),FaceBallReward(),TouchBallReward(),VelocityBallToGoalReward(),EventReward()
(0.4, 0.1, 0.6, 0.4, 1, 1)
50 sec frames
ca 400 rew
Kørte i 10 timer, virkede til at FaceBall tog overhånd

14-01-2023
VelocityPlayerToBallReward(),VelocityReward(),TouchBallReward(),VelocityBallToGoalReward(),EventReward()
(0.8, 0.1, 0.6, 0.5, 1)

30000 lines = 8300000

Har nok haft 350_000_000 steps mindst inden jeg fik slået det fra hvor den resetter steps

950.000.000 ticks = 63.333.333 sec = 1.055.556 min = 17.593 hours
'''

