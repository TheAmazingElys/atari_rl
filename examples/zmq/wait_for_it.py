import zmq, time, subprocess
from multiprocessing import Process
import setproctitle

setproctitle.setproctitle("Python: waiting_for_it")

def call_is_waiting_for_it():
    args = ["python",
        "is_waiting_for_it.py"]
    subprocess.call(args)


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

socket_REP = context.socket(zmq.REP)
socket_REP.bind("tcp://*:5557")

p = Process(target=call_is_waiting_for_it)
p.start()

time.sleep(1)
for i in range(10):
    print(f"Sending {i}")
    socket.send_string(f"{i}")
    message = socket_REP.recv_string()
    print(f"{i} confirmed!")
    socket_REP.send_string("OK!")

p.join()

