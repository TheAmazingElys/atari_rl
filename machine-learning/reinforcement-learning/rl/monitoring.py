import numpy as np

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