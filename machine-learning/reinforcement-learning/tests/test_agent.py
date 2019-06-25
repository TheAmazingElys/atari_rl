from rl.agents import DQNParameters, DoubleDQN
from rl.brain import AtariDQN
from rl.environment import AtariEnvironment, EnvNames
from rl.game import Game

def test_agent():

    env = AtariEnvironment(env_name=EnvNames.SPACE_INVADER, render=False)
    brain = AtariDQN(env.get_number_of_actions())    
    agent = DoubleDQN(brain)
    game = Game(agent, env)
    game.run(horizon=1)

    assert True # If nothing crashed we are happy