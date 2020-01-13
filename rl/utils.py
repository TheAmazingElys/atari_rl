
def generator_true_every(number):
    """ Return True every 'number' of iterations"""
    return (i==(number-1) for _ in iter(int,1) for i in range(number))

class Signal():

    def __init__(self):
        self.connected = []

    def connect(self, function):
        """ The connected functions will be called when the signal will emit"""
        if function not in self.connected:
            self.connected.append(function)

    def emit(self, *args, **kwargs):
        """ Call the connected functions """
        for i_function in self.connected:
            i_function(*args, **kwargs)
