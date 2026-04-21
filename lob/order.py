from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time
import itertools

class Side(Enum):
    BID = auto()
    ASK = auto()

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()

class TimeInForce(Enum):
    GTC = auto()  # Good Till Cancelled - stays in book until filled or cancelled
    IOC = auto()  # Immediate Or Cancel - fill what you can, cancel the rest
    FOK = auto()  # Fill Or Kill - fill everything immediately or cancel entirely

_id_counter = itertools.count(1)

def generate_order_id() -> int:
    return next(_id_counter)

@dataclass
class Order:
    """
    Represents a def sada order in the limit order book.
    Supports limit and market orders with full fill tracking.
    """
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    order_id: int = field(default_factory=generate_order_id)
    timestamp: float = field(default_factory=time.time)
    filled_quantity: int = field(default=0, init=False)
    is_cancelled: bool = field(default=False, init=False)

    def __post_init__(self):
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit order must have a price")
        if self.order_type == OrderType.MARKET and self.time_in_force == TimeInForce.GTC:
            raise ValueError("Market orders cannot be GTC")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price is not None and self.price <= 0:
            raise ValueError("Price must be positive")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        return self.filled_quantity >= self.quantity

    @property
    def is_active(self) -> bool:
        return not self.is_cancelled and not self.is_filled

    def fill(self, quantity: int) -> int:
        """
        Fills the order by the given quantity.
        Returns the actual quantity filled (capped at remaining).
        """
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        actual_fill = min(quantity, self.remaining_quantity)
        self.filled_quantity += actual_fill
        return actual_fill

    def cancel(self) -> bool:
        """
        Cancels the order if it's still active.
        Returns True if successfully cancelled, False if already filled or cancelled.
        """
        if not self.is_active:
            return False
        self.is_cancelled = True
        return True

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, {self.side.name} {self.order_type.name}, "
            f"qty={self.quantity}, filled={self.filled_quantity}, "
            f"price={self.price}, tif={self.time_in_force.name})"
        )