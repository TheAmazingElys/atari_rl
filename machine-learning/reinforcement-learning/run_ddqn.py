from rl.agents import DoubleDQN, DQNParameters
from rl.brain import DQN, AtariDQN
from rl.environment import get_env_and_input_layer, EnvNames, AtariEnvironment
from rl.game import Game
from rl.monitoring import Monitor


if __name__ == "__main__":
    """
    env, input_net = get_env_and_input_layer(env_name=EnvNames.POLECART, render=False)
    param = DQNParameters(capacity=40000, waiting_time = 20000, lr = 1E-4, frozen_steps=1000, gamma=0.9)
    brain = DQN(input_net, env.get_number_of_actions(), 16)
    agent = DoubleDQN(brain, param)
    game = Game(agent, env)

    monitor = Monitor(agent, game)

    game.run(100000)
    """
    env = AtariEnvironment(env_name=EnvNames.SPACE_INVADER, render=False)
    brain = AtariDQN(env.get_number_of_actions())    
    agent = DoubleDQN(brain)
    game = Game(agent, env)

    monitor = Monitor(agent, game)

    game.run(horizon=10000000)
