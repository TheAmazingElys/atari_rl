import torch
    
def test_convolution():

    from rl.brain import AtariConvolution
    conv = AtariConvolution()
    tensor = torch.rand([1, 4, 84, 84])

    result = conv(tensor)

    assert result.shape == (1, 64, 10, 10)
    assert result.data.numpy().size == conv.size