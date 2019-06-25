import copy, numpy, torch
import torch.optim as optim
import torch.nn.functional as F

from rl.brain import AtariDQN
from rl.memory import PrioritizedMemory, Transition
from rl.utils import generator_true_every, Signal

class RandomAgent():
    """
    Random agent
    """

    def __init__(self, actions = 1):
        self.actions = actions
        
    def select_action(self, *args, **kwargs):
        return numpy.random.randint(self.actions)
        
    def observe(self, *args, **kwargs): 
        """
        Observe an experience tuple (state, action, reward, resulting_state)
        """
        pass

class DQNParameters():
    def __init__(self, clipping = 1, capacity = 100000, gamma = 0.99, lr = 0.0000625, batch_size = 32, replay_period = 4, frozen_steps = 32000, waiting_time = 80000):
        """
        clipping : clipping of reward
        capacity : the capacity of the memory
        frozen_steps : the number of steps before updating the frozen network
        """
        self.clipping = clipping
        self.gamma = gamma
        self.lr = lr
        self.capacity = capacity
        self.batch_size = batch_size
        self.replay_period = replay_period
        self.frozen_steps = frozen_steps
        self.waiting_time = waiting_time

class DoubleDQN():
    """
    From Deep Reinforcement Learning with Double Q-learning
    at https://arxiv.org/abs/1509.06461
    """

    def __init__(self, DQN, parameters = DQNParameters()):
        """
        DQN: The DQN used to estimate the reward
        parameters: The parameters!
        """

        self.on_loss_computed = Signal()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.DQN = DQN.to(self.device).train()

        self.frozen_DQN = copy.deepcopy(self.DQN).eval()
        for param in self.frozen_DQN.parameters():
            param.requires_grad = False
        self._update_frozen()

        self.memory = PrioritizedMemory(parameters.capacity)

        self.optimizer = optim.Adam(self.DQN.parameters(), lr = parameters.lr)
        self.parameters = parameters

        self.it_s_replay_time = generator_true_every(self.parameters.batch_size)
        self.it_s_update_frozen_time = generator_true_every(self.parameters.frozen_steps)

        self.it_s_action_debug_time = generator_true_every(1000)

    def _update_frozen(self):
        """
        Let it go, let it go
        I am one with the wind and sky
        Let it go, let it go
        You'll never see me cry
        Here I stand and here I stay
        Let the storm rage on
        """
        self.frozen_DQN.load_state_dict(self.DQN.state_dict())

    def select_action(self, state):
        """ Return the selected action """
        with torch.no_grad():
            self._reset_noise()
            values = self.DQN(torch.FloatTensor([state]).to(self.device)).cpu().data.numpy()[0]
            if self.memory.total() > self.parameters.waiting_time:
                selected_action = numpy.argmax(values)
                if next(self.it_s_action_debug_time):
                    print(selected_action, values)
            else:
                selected_action = numpy.random.randint(len(values))

            return selected_action

    def observe(self, state, action, reward, next_state, is_terminal): 
        """
        Observe an experience tuple (state, action, reward, next_state, is_terminal)
        """

        if self.parameters.clipping is not None: # Clip the reward
            reward = numpy.clip(reward, -self.parameters.clipping, self.parameters.clipping)
        
        self.memory.add(10, state, action, reward, next_state, is_terminal)
    
        if next(self.it_s_update_frozen_time):
            self._update_frozen()
        
        if next(self.it_s_replay_time) and self.memory.total() > self.parameters.waiting_time/2:
            self._replay()

    def train(self):
        self.DQN.train()
        
    def eval(self):
        self.DQN.eval()

    def save(self):
        self.DQN.save_state_dict("model.torch")

    def _reset_noise(self):
        if hasattr(self.DQN, "reset_noise"):
            self.DQN.reset_noise()

    def _replay(self):
        """
        Learn things
        """
        indexes, transitions = zip(*self.memory.sample(self.parameters.batch_size))
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        self._reset_noise()

        state_values = self.DQN(\
            torch.FloatTensor(batch.state).to(self.device),\
            torch.LongTensor(batch.action).to(self.device).unsqueeze(1)\
        )
        with torch.no_grad():
            expected_state_values = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)\
                + self.parameters.gamma * self.DQN(torch.FloatTensor(batch.next_state).to(self.device)).max(1, True)[0]*(1 - torch.FloatTensor(batch.terminal).to(self.device).unsqueeze(1))

        loss = F.smooth_l1_loss(state_values, expected_state_values)# Huber Loss
        self.on_loss_computed.emit(loss.cpu().data.numpy()) # Emit the computed loss

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()