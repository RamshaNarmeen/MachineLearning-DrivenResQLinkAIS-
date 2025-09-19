
import asyncio
import math
import random
import time
from typing import Dict, List, Optional

from messages import Message, MsgType
from lml_model import LMLModel

class Link:
    def __init__(self, a: "Node", b: "Node", base_loss=0.1):
        self.a = a
        self.b = b
        self.base_loss = base_loss  # packet loss baseline
        self.queue_delay = 0.02     # seconds per queued item

    def stats_for(self, src: "Node", dst: "Node"):
        # simple symmetric stats with noise
        rssi = -40 - 5*random.random() if src.distance_to(dst) < 1.0 else -70 - 10*random.random()
        loss = min(0.9, self.base_loss + 0.05*random.random())
        qlen = len(dst._in_queue)
        dist = src.distance_to(dst)
        return {'rssi': rssi, 'loss': loss, 'queue': qlen, 'distance_to_dst': dist}

class Node:
    def __init__(self, node_id: str, x: float, y: float):
        self.id = node_id
        self.x = x
        self.y = y
        self.links: Dict[str, Link] = {}  # neighbor_id -> Link
        self.model = LMLModel(node_id=self.id)

        self._in_queue: List[Message] = []
        self._out_stats = {'in_count':0,'out_count':0,'fail_count':0,'attempts':0,'queue_len':0}
        self._delivered: List[Message] = []
        self._running = False

    def connect(self, other: "Node", base_loss=0.1):
        link = Link(self, other, base_loss=base_loss)
        self.links[other.id] = link
        other.links[self.id] = link

    def distance_to(self, other: "Node"):
        return ((self.x-other.x)**2 + (self.y-other.y)**2)**0.5

    def receive(self, msg: Message):
        self._in_queue.append(msg)

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(0.01)
            self._out_stats['queue_len'] = len(self._in_queue)
            # process a limited number per tick to simulate CPU
            for _ in range(min(5, len(self._in_queue))):
                msg = self._in_queue.pop(0)
                self._out_stats['in_count'] += 1
                await self.process_message(msg)

            # simple anomaly monitoring
            a = self.model.anomaly_score(self._out_stats)
            if a > 0.8:
                # react: drop low-priority for a short period
                pass
            # decay counts
            for k in ['in_count','out_count','fail_count','attempts']:
                self._out_stats[k] = int(self._out_stats[k]*0.8)

    async def process_message(self, msg: Message):
        # drop duplicates in path
        if self.id in msg.path:
            return
        msg.path.append(self.id)

        # classification & priority (single model, simultaneous in concept)
        mtype, conf = self.model.classify(msg)
        msg.mtype = mtype
        if mtype == MsgType.DISTRESS and conf > 0.7:
            msg.meta['tag_distress'] = True
        msg.priority = self.model.prioritize(msg, {'queue_len': len(self._in_queue)})

        # delivery condition
        if msg.dst is None or msg.dst == self.id:
            # delivered to this node (responder or broadcast target)
            self._delivered.append(msg)
            return

        # pick next hop
        next_hops = {}
        for nid, link in self.links.items():
            neigh = link.a if link.b.id == self.id else link.b
            next_hops[nid] = link.stats_for(self, neigh)

        nxt = self.model.route(next_hops, {'queue_len': len(self._in_queue)}, msg)
        if nxt is None:
            return

        # simulate send
        self._out_stats['attempts'] += 1
        link = self.links[nxt]
        neigh = link.a if link.b.id == self.id else link.b
        # random loss
        if random.random() < link.base_loss:
            self._out_stats['fail_count'] += 1
            return

        # enqueue to neighbor
        msg.hops += 1
        neigh.receive(msg)
        self._out_stats['out_count'] += 1

    def stop(self):
        self._running = False

    def delivered(self):
        return list(self._delivered)
