import imageio, numpy
from rl.utils import Signal, generator_true_every

class Game():

    def __init__(self, agent, environment): 

        self.environment = environment
        self.agent = agent
        self.done = True
        self.cumulative_reward = None
        self.on_game_ended = Signal()
        self.it_s_test_time = generator_true_every(10)

    def __init_new_game__(self):

        self.cumulative_reward = 0
        self.current_state = self.environment.reset()
        self.done = False

    def run(self, horizon = 1, gif_path = None, gif_in_color = False):

        t = 0
        test_time = False
                    
        while t<horizon or not self.done: # Wait for the current game to end when t>=horizon

            if self.done:
                self.__init_new_game__() # Initialisation of the new game
                if next(self.it_s_test_time):
                    if hasattr(self.agent, "eval"):
                        self.agent.eval()
                    test_time = True
                else:
                    if hasattr(self.agent, "train"):
                        self.agent.train()
                    test_time = False

            t+=1

            action = self.agent.select_action(self.current_state)
            
            next_state, reward, self.done, _ = self.environment.make_step(action)
            self.cumulative_reward += reward
                
            if self.current_state is not None:
                self.agent.observe(self.current_state, action, reward, next_state, self.done)

            self.current_state = next_state
            
            if self.done:
                self.on_game_ended.emit(self.cumulative_reward)