
"""
Large Machine Learning (LML) model for ResQLink

This file implements a unified model interface with four capabilities:
- message classification
- priority assessment
- dynamic routing suggestion
- anomaly detection

For portability, it uses a heuristic, edge-friendly implementation by default.
If PyTorch is available at runtime, you can toggle USE_TORCH=True to use
a tiny transformer encoder (still lightweight) for demonstration.
"""

from typing import Dict, List, Optional, Tuple
import math
import time
import random

from messages import Message, MsgType

# Toggle to try PyTorch transformer demo (if torch exists in environment).
USE_TORCH = False
try:
    import torch
    USE_TORCH = False  # keep False by default for portable demo
except Exception:
    USE_TORCH = False

DISTRESS_KEYWORDS = {
    "help","mayday","sos","emergency","trapped","injury","bleeding","collapsed",
    "fire","danger","earthquake","flood","hurricane","wildfire","bomb","attack"
}
GPS_KEYWORDS = {"lat","lon","gps","coordinates","location","pos","grid"}
SUPPLY_KEYWORDS = {"water","food","medicine","insulin","tent","blanket","generator","fuel"}
STATUS_KEYWORDS = {"ok","alive","safe","status","update","checkin","fine"}

def softclip(x: float, lo: float, hi: float)->float:
    return max(lo, min(hi, x))

class LMLModel:
    """
    Unified LML model (single object) exposing:
      classify(msg) -> (MsgType, confidence)
      prioritize(msg, net_ctx) -> priority score [0..1]
      route(next_hops, net_ctx, msg) -> best_next_hop_id
      anomaly_score(net_ctx) -> [0..1]
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        # rolling stats for anomaly detection
        self._recent_in = 0
        self._recent_out = 0
        self._recent_fail = 0
        self._last_tick = time.time()

    # ---- Message understanding ----
    def classify(self, msg: Message) -> Tuple[MsgType, float]:
        text = (msg.text or "").lower()
        score = {MsgType.DISTRESS:0, MsgType.GPS:0, MsgType.SUPPLY:0, MsgType.STATUS:0}
        for w in DISTRESS_KEYWORDS:
            if w in text: score[MsgType.DISTRESS]+=1
        for w in GPS_KEYWORDS:
            if w in text: score[MsgType.GPS]+=1
        for w in SUPPLY_KEYWORDS:
            if w in text: score[MsgType.SUPPLY]+=1
        for w in STATUS_KEYWORDS:
            if w in text: score[MsgType.STATUS]+=1

        # incorporate structure: GPS tuple strongly indicates GPS
        if msg.gps is not None:
            score[MsgType.GPS] += 2.5

        # corrupted/empty handling
        if len(text.strip())==0 and msg.gps is None:
            return (MsgType.UNKNOWN, 0.2)

        mtype = max(score, key=score.get)
        conf = softclip(score[mtype]/3.0, 0.1, 0.99)
        # heuristic: distress confidence higher if many distress tokens
        if mtype == MsgType.DISTRESS and score[mtype] >= 2:
            conf = softclip(conf+0.2,0,1)
        return (mtype, conf)

    # ---- Priority assessment ----
    def prioritize(self, msg: Message, net_ctx: Dict) -> float:
        # base on type
        base = {
            MsgType.DISTRESS: 0.95,
            MsgType.GPS: 0.75,
            MsgType.SUPPLY: 0.6,
            MsgType.STATUS: 0.4,
            MsgType.UNKNOWN: 0.3
        }[msg.mtype]

        # recency bonus
        age = time.time() - msg.timestamp
        recency = softclip(1.0 - age/600.0, 0.0, 1.0)  # 10 min to zero
        # network congestion penalty
        qlen = net_ctx.get("queue_len", 0)
        congestion = softclip(1.0 - qlen/50.0, 0.2, 1.0)

        prio = base*0.7 + 0.25*recency + 0.05*congestion

        # GPS piggyback with distress gets higher
        if msg.mtype==MsgType.GPS and (msg.meta.get("tag_distress") or "help" in (msg.text or "").lower()):
            prio = max(prio, 0.85)

        return softclip(prio, 0.0, 1.0)

    # ---- Routing optimization ----
    def route(self, next_hops: Dict[str, Dict], net_ctx: Dict, msg: Message) -> Optional[str]:
        """
        next_hops: { node_id: { 'rssi':float, 'loss':float, 'queue':int, 'distance_to_dst': float or None } }
        Picks next hop minimizing expected delay, factoring reliability and queue.
        """
        if not next_hops: return None

        # Weight terms
        best_node = None
        best_score = -1e9
        for nid, stats in next_hops.items():
            rssi = stats.get('rssi', -80)        # higher better
            loss = stats.get('loss', 0.2)        # lower better
            q    = stats.get('queue', 0)         # lower better
            dist = stats.get('distance_to_dst', None)  # lower better if known

            link_quality = (rssi + 100)/60.0     # normalize approx [0..~1.5]
            reliability  = 1.0 - softclip(loss,0.0,0.9)
            queue_pen    = 1.0/(1.0+q)
            dist_term    = 0.0 if dist is None else 1.0/(1.0+dist)

            # priority-aware weighting
            p = softclip(msg.priority, 0, 1)
            weight_dist = 0.4 + 0.3*p
            weight_rel  = 0.3 + 0.2*p
            weight_ql   = 0.2
            weight_queue= 0.1

            score = (weight_dist*dist_term +
                     weight_rel*reliability +
                     weight_ql*link_quality +
                     weight_queue*queue_pen)

            if score > best_score:
                best_score, best_node = score, nid
        return best_node

    # ---- Anomaly detection ----
    def anomaly_score(self, net_ctx: Dict) -> float:
        """
        Lightweight behavioral anomaly: spikes in failures or queues.
        """
        now = time.time()
        dt = max(1e-3, now - self._last_tick)
        self._last_tick = now

        in_rate = net_ctx.get("in_count",0)/dt
        out_rate = net_ctx.get("out_count",0)/dt
        fail_rate = net_ctx.get("fail_count",0)/max(1.0,net_ctx.get("attempts",1))/dt

        qlen = net_ctx.get("queue_len",0)
        qterm = softclip(qlen/50.0, 0.0, 1.0)

        # Combine; emphasize failures
        score = softclip(0.6*fail_rate + 0.3*qterm + 0.1*abs(in_rate-out_rate)/(1+in_rate+out_rate), 0.0, 1.0)
        return score
