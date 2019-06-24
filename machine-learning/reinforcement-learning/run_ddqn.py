from rl.agents import DQNParameters, DoubleDQN
from rl.brain import AtariDQN
from rl.environment import Environment, EnvNames
from rl.game import Game
from rl.monitoring import Monitor


if __name__ == "__main__":

    env = Environment(env_name=EnvNames.SPACE_INVADER.value, render=False)
    brain = AtariDQN(env.get_number_of_actions())    
    agent = DoubleDQN(brain)
    game = Game(agent, env)

    monitor = Monitor(agent, game)

    game.run(horizon=10000000)