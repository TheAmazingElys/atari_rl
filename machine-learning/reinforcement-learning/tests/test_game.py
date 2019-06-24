from rl.agents import RandomAgent
from rl.environment import Environment, EnvNames
from rl.game import Game
from rl.monitoring import Monitor



def test_game():

    env = Environment(env_name=EnvNames.SPACE_INVADER.value, render=False)
    agent = RandomAgent(env.get_number_of_actions())
    game = Game(agent, env)

    monitor = Monitor(agent, game)

    game.run(gif_path='tests/samples/spaceinvader.gif')

    assert len(monitor.reward) > 0