from rl.brain import AtariDQN
import torch, numpy as np
    
def test_advantage_value():

    value = torch.FloatTensor([[0],[0]])
    advantage = torch.FloatTensor([[0,1,2],[0,1,2]])
    result = AtariDQN.compute_advantage_value(advantage, value)
    expected_result = np.array([[-1.,  0.,  1.], [-1.,  0.,  1.]])
    assert (result.data.numpy() == expected_result).all()

    assert (AtariDQN.gather(result).data.numpy() == expected_result).all()

    indexes = torch.LongTensor([[1],[0]])
    expected_result = np.array([[0.],[-1.]])
    assert (AtariDQN.gather(result, indexes).data.numpy() == expected_result).all()
        