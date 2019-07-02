import numpy as np
from rl.game import Game
from rl.environment import EnvNames
from PIL import Image

class GifCreator():

    def __init__(self, game : Game):

        self.states = []
        self.game = game
        game.on_game_ended.connect(self.save_gif)
        game.environment.on_new_state.connect(self.append)

    def append(self, state):
        self.states.append(state)

    def save_gif(self, reward):
        path = f"gif/{str(int(reward))}_{self.game.environment.name}.gif"

        self.states = [Image.fromarray(i_state) for i, i_state in enumerate(self.states) if i%3 ==0]
        self.states[0].save(path, format='GIF', append_images=self.states[1:], save_all=True, duration=1, loop=0)
        self.states = []



class Monitor():

    def __init__(self, agent, game):
        """
        Connect to the Agent 
        """
        self.reward = []
        self.loss = []
        self.best_reward = 0

        self.agent = agent
        self.game = game

        game.on_game_ended.connect(self.log_reward)
        if hasattr(agent, "on_loss_computed"):
            agent.on_loss_computed.connect(self.log_loss)

    def log_reward(self, reward):
        self.reward.append(reward)

        if self.best_reward < reward:
            self.best_reward = reward
            if hasattr(self.agent, "save"):
                self.agent.save()

        if len(self.loss) > 0:
            print(f"Reward: {reward}, Loss: {np.mean(self.loss)}")
            self.loss.clear()

    def log_loss(self, loss):
        self.loss.append(loss)