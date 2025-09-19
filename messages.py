
from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Any, Dict, Optional

class MsgType(Enum):
    DISTRESS = auto()
    GPS = auto()
    STATUS = auto()
    SUPPLY = auto()
    UNKNOWN = auto()

@dataclass
class Message:
    src: str
    dst: Optional[str]  # None means broadcast to responders
    text: str
    gps: Optional[tuple] = None  # (lat, lon)
    timestamp: float = field(default_factory=lambda: time.time())
    id: str = field(default_factory=lambda: f"msg-{int(time.time()*1000)}")
    meta: Dict[str, Any] = field(default_factory=dict)

    # runtime fields used by the mesh
    priority: float = 0.0
    mtype: MsgType = MsgType.UNKNOWN
    hops: int = 0
    path: list = field(default_factory=list)

    def __repr__(self):
        return f"<Message {self.id} {self.mtype.name} prio={self.priority:.2f} hops={self.hops} from {self.src} to {self.dst or 'ANY'}>"
