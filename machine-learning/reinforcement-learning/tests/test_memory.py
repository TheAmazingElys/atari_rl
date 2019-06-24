from rl.memory import Transition, PrioritizedMemory


def test_memory():

    memory = PrioritizedMemory(10)
    memory.add(15, 1, 2, 3, 4, 5)
    memory.add(10, 4, 5, 6, 5, 2)

    indexes, transitions = zip(*memory.sample(2))

    assert indexes == (9,10)
    assert transitions == (Transition(state=1, action=2, next_state=3, reward=4, terminal=5), Transition(state=4, action=5, next_state=6, reward=5, terminal=2))

    """ Example of batch creation """
    assert Transition(*zip(*transitions))  == Transition(state=(1, 4), action=(2, 5), next_state=(3, 6), reward=(4, 5), terminal=(5, 2))