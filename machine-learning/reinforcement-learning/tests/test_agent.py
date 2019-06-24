from rl.agents import DQNParameters, DoubleDQN
from rl.brain import AtariDQN
from rl.environment import Environment, EnvNames
from rl.game import Game

def test_agent():

    env = Environment(env_name=EnvNames.SPACE_INVADER.value, render=True)
    brain = AtariDQN(env.get_number_of_actions())    
    agent = DoubleDQN(brain)
    game = Game(agent, env)
    game.run(horizon=1000)

    assert True # If nothing crashed we are happy