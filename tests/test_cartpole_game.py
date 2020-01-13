from rl.agents import DoubleDQN, DQNParameters
from rl.brain import DQN
from rl.environment import get_env_and_input_layer, EnvNames
from rl.game import Game
from rl.monitoring import Monitor


def test_cartpole_game():

    env, input_net = get_env_and_input_layer(env_name=EnvNames.POLECART, render=False)
    param = DQNParameters(capacity=1000, waiting_time = 1000)
    brain = DQN(input_net, env.get_number_of_actions())
    agent = DoubleDQN(brain, param)
    game = Game(agent, env)

    monitor = Monitor(agent, game)

    game.run(1)