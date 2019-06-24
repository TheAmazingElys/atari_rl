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
    def __init__(self, clipping = 1, capacity = 100000, gamma = 0.99, lr = 1E-4, batch_size = 32, frozen_steps = 1000):
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
        self.frozen_steps = frozen_steps

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
        self.DQN = DQN.to(self.device)
        self.memory = PrioritizedMemory(parameters.capacity)
        self._update_frozen()

        self.optimizer = optim.Adam(self.DQN.parameters())
        self.parameters = parameters

        self.it_s_batch_time = generator_true_every(self.parameters.batch_size)
        self.it_s_update_frozen_time = generator_true_every(self.parameters.frozen_steps)
        

    def _update_frozen(self):
        """
        Let it go, let it go
        I am one with the wind and sky
        Let it go, let it go
        You'll never see me cry
        Here I stand and here I stay
        Let the storm rage on
        """
        self.frozen_DQN = copy.deepcopy(self.DQN)
        self.frozen_DQN.to(self.device).eval()


    def select_action(self, state):
        """ Return the selected action """
        return numpy.argmax(self.DQN(torch.FloatTensor([state], device = self.device)).data.numpy()[0])


    def observe(self, state, action, reward, next_state, is_terminal): 
        """
        Observe an experience tuple (state, action, reward, next_state, is_terminal)
        """

        if self.parameters.clipping is not None: # Clip the reward
            reward = numpy.clip(reward, -self.parameters.clipping, self.parameters.clipping)
        
        """ 
        Compute the error to update memory weights 
        Tensors are unsqueezed because the DQN requires a batch as input

        """
        Q_value = self.DQN(torch.FloatTensor(state).unsqueeze(0), torch.LongTensor(numpy.array(action).reshape(1,-1)))[0].data.numpy()
        next_Q_value = self.parameters.gamma * numpy.amax(self.DQN(torch.FloatTensor(next_state).unsqueeze(0))[0].data.numpy()) if not is_terminal else 0
        error = abs(reward + next_Q_value - Q_value)

        self.memory.add(error, state, action, reward, next_state, is_terminal)
    
        if next(self.it_s_update_frozen_time):
            self._update_frozen()
        
        if next(self.it_s_batch_time):
            self._replay()

    def train(self):
        self.DQN.train()
        
    def eval(self):
        self.DQN.eval()
        
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

        state_values = self.DQN(\
            torch.FloatTensor(batch.state, device = self.device),\
            torch.LongTensor(batch.action, device = self.device).unsqueeze(1)\
        )
        try:
            expected_state_values = torch.FloatTensor(batch.reward, device = self.device).unsqueeze(1)\
                + self.parameters.gamma * self.DQN(torch.FloatTensor(batch.next_state, device = self.device)).max(1, True)[0]*(1 - torch.FloatTensor(batch.terminal, device = self.device).unsqueeze(1))
        except:
            import pdb; pdb.set_trace()
            
        loss = F.smooth_l1_loss(state_values, expected_state_values)# Huber Loss
        self.on_loss_computed.emit(loss.data.numpy()) # Emit the computed loss

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self._reset_noise()
