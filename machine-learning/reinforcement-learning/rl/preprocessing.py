import cv2, numpy

def transform_state(state):
    """ Convert a state to grayscale and crop the scores """
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = state[24:204,:]
    state = cv2.resize(state, (84,84))
    state = state[:,:,]
    return state

class ProcessBuffer():
    
    def __init__(self, size, transform = lambda x : x):
        
        self.size = size
        self.buffer = []
        self.transform = transform
        
    def add_state(self, state):
        """ Add a state to the buffer and apply the transform function """

        if not getattr(self, "buffer"):
            self.reinit_buffer(state)

        self.buffer = self.buffer[1:]
        self.buffer.append(self.transform(state))
            
    def get_processed_state(self, state = None):
        """ 
        Return the preprocessed state or the last preprocessed state in the buffer if None.
        The buffer is not modified by this operation.
        """

        if not getattr(self, "buffer"):
            assert False, "Call reinit_buffer first"

        if state: 
            return numpy.stack(self.buffer[1:] + [self.transform(state)])
        else:
            return numpy.stack(self.buffer)

    
    def reinit_buffer(self, p_state):
        """ Clear the buffer """

        p_state = self.transform(p_state)
        self.buffer = [p_state]*self.size
        