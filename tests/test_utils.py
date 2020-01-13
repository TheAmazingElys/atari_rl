from rl.utils import generator_true_every

def test_generator_true_every_number():

    generator = generator_true_every(3)

    for i in range(2):
        assert not next(generator)
        assert not next(generator)
        assert next(generator)

