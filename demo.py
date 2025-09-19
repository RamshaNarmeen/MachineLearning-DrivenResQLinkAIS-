
import asyncio
import random
import time

from messages import Message, MsgType
from mesh import Node

async def main():
    # Create nodes
    a = Node("A", 0.0, 0.0)
    b = Node("B", 0.8, 0.0)
    c = Node("C", 1.6, 0.2)
    d = Node("D", 2.4, 0.0)  # responder

    # Connect nodes (mesh)
    a.connect(b, base_loss=0.05)
    b.connect(c, base_loss=0.08)
    c.connect(d, base_loss=0.12)
    # also a diagonal link with higher loss
    b.connect(d, base_loss=0.25)

    # Start nodes
    tasks = [asyncio.create_task(n.run()) for n in [a,b,c,d]]

    # Send mixed messages into the mesh
    msgs = [
        Message(src="A", dst="D", text="HELP trapped under debris near school", gps=(40.1, -74.2)),
        Message(src="B", dst="D", text="status ok at clinic, minor injuries"),
        Message(src="C", dst="D", text="need water and medicine at shelter"),
        Message(src="A", dst="D", text="GPS coordinates lat 40.12 lon -74.25", gps=(40.12,-74.25)),
        Message(src="B", dst="D", text="smoke and fire spreading to east"),
    ]

    for m in msgs:
        # device-to-device handoff: inject at source
        if m.src == "A":
            a.receive(m)
        elif m.src == "B":
            b.receive(m)
        elif m.src == "C":
            c.receive(m)

    # allow time for delivery
    await asyncio.sleep(2.0)

    # Stop nodes
    for n in [a,b,c,d]:
        n.stop()
    await asyncio.gather(*tasks)

    # Print delivery results
    print("Delivered at D:")
    for m in d.delivered():
        print(f" - {m} | path={m.path}")

if __name__ == "__main__":
    asyncio.run(main())
