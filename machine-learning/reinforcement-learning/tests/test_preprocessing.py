from rl.preprocessing import ProcessBuffer, transform_state
import gym, imageio, json

def test_preprocessing():

    buffer = ProcessBuffer(2)
    buffer.add_state([0,1])

    assert (buffer.get_processed_state() == [[0,1],[0,1]]).all()
    assert (buffer.get_processed_state([2,1]) == [[0,1],[2,1]]).all()

    buffer.add_state([3,0])
    assert (buffer.get_processed_state([2,1]) == [[3,0],[2,1]]).all()

    buffer.reinit_buffer([1,1])
    assert (buffer.get_processed_state() == [[1,1],[1,1]]).all()


def test_img():

    env = gym.make('SpaceInvadersNoFrameskip-v4')
    _ = env.reset()

    for i in range(500):
        state = env.step(2)

    initial_state = state[0]
    imageio.imwrite('tests/samples/spaceinvader.png', initial_state)
    processed_state = transform_state(initial_state)
    imageio.imwrite('tests/samples/spaceinvader_processed.png', processed_state)

    shapes_dict = {"initial_shape" : initial_state.shape, "processed_shape" : processed_state.shape}

    with open('tests/samples/atari_shapes.json', 'w') as outfile:  
        json.dump(shapes_dict, outfile)
