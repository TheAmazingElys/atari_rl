import zmq
from timeit import default_timer as timer

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

socket.connect("tcp://127.0.0.1:5556")
socket.subscribe = ''

socket_REQ = context.socket(zmq.REQ)
socket_REQ.connect("tcp://127.0.0.1:5557")

start = timer()
while True:
    if timer() - start > 10:
        print("timeout")
        break

    if socket.poll(100) == zmq.POLLIN:
        string = socket.recv_string()
        print(f"Receiving {string}")
        print(f"Sending confirmation!")
        socket_REQ.send_string("Confirmation")
        _ = socket_REQ.recv_string()

