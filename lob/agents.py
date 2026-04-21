import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .order import Order, Side, OrderType, TimeInForce
from .book import LimitOrderBook

rng = np.random.default_rng()

@dataclass
class AgentConfig:
    """
    Base configuration shared across all agent types.
    arrival_rate: average number of orders per second (Poisson process)
    """
    arrival_rate: float
    min_quantity: int = 1
    max_quantity: int = 100

class BaseAgent:
    """
    Abstract base for all market participants.
    Each agent has a unique ID and acts on a shared order book.
    """
    _id_counter: int = 0

    def __init__(self, config: AgentConfig, book: LimitOrderBook):
        BaseAgent._id_counter += 1
        self.agent_id = BaseAgent._id_counter
        self.config = config
        self.book = book
        self.active_orders: dict[int, Order] = {}
        self.pnl: float = 0.0
        self.inventory: int = 0  # positive = long, negative = short

    def _sample_quantity(self) -> int:
        return int(rng.integers(self.config.min_quantity, self.config.max_quantity + 1))

    def _next_arrival_time(self) -> float:
        """Inter-arrival time drawn from exponential distribution (Poisson process)."""
        return rng.exponential(1.0 / self.config.arrival_rate)

    def _update_inventory(self, side: Side, quantity: int, price: float) -> None:
        if side == Side.BID:
            self.inventory += quantity
            self.pnl -= quantity * price
        else:
            self.inventory -= quantity
            self.pnl += quantity * price

    def act(self) -> list[Order]:
        raise NotImplementedError


class MarketMaker(BaseAgent):
    """
    Continuously quotes both sides of the book around the mid price.
    Skews quotes based on inventory to manage risk — this is the
    Avellaneda-Stoikov market making model simplified.

    When long, lowers both bid and ask to encourage selling.
    When short, raises both to encourage buying.
    """

    def __init__(self, config: AgentConfig, book: LimitOrderBook,
                 spread: float = 0.10,
                 max_inventory: int = 500,
                 skew_factor: float = 0.01):
        super().__init__(config, book)
        self.spread = spread
        self.max_inventory = max_inventory
        self.skew_factor = skew_factor

    def _compute_quotes(self) -> tuple[float, float]:
        """
        Returns (bid_price, ask_price) adjusted for inventory skew.
        Skew pushes quotes away from the direction of current inventory.
        """
        mid = self.book.mid_price
        if mid is None:
            mid = 100.0  # default reference price if book is empty

        skew = self.skew_factor * self.inventory
        half_spread = self.spread / 2

        bid_price = round(mid - half_spread - skew, 2)
        ask_price = round(mid + half_spread - skew, 2)

        bid_price = max(bid_price, 0.01)
        ask_price = max(ask_price, bid_price + 0.01)

        return bid_price, ask_price

    def _cancel_stale_orders(self) -> None:
        """Cancel all resting orders before requoting."""
        for order_id in list(self.active_orders.keys()):
            self.book.cancel(order_id)
        self.active_orders.clear()

    def act(self) -> list[Order]:
        """
        Cancel existing quotes and post fresh ones.
        Skips quoting if inventory limit is breached on one side.
        """
        self._cancel_stale_orders()
        bid_price, ask_price = self._compute_quotes()
        qty = self._sample_quantity()
        orders = []

        if self.inventory < self.max_inventory:
            bid = Order(
                side=Side.BID,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=bid_price,
                time_in_force=TimeInForce.GTC,
            )
            trades = self.book.submit(bid)
            self.active_orders[bid.order_id] = bid
            for t in trades:
                self._update_inventory(Side.BID, t.quantity, t.price)
            orders.append(bid)

        if self.inventory > -self.max_inventory:
            ask = Order(
                side=Side.ASK,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=ask_price,
                time_in_force=TimeInForce.GTC,
            )
            trades = self.book.submit(ask)
            self.active_orders[ask.order_id] = ask
            for t in trades:
                self._update_inventory(Side.ASK, t.quantity, t.price)
            orders.append(ask)

        return orders


class NoiseTrader(BaseAgent):
    """
    Uninformed trader that submits random orders.
    Represents retail flow with no directional signal.
    Mix of market and limit orders with random sizes and prices.
    """

    def __init__(self, config: AgentConfig, book: LimitOrderBook,
                 market_order_prob: float = 0.3,
                 limit_offset_range: float = 0.05):
        super().__init__(config, book)
        self.market_order_prob = market_order_prob
        self.limit_offset_range = limit_offset_range

    def act(self) -> list[Order]:
        side = rng.choice([Side.BID, Side.ASK])
        qty = self._sample_quantity()
        use_market = rng.random() < self.market_order_prob

        if use_market:
            order = Order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
            )
        else:
            mid = self.book.mid_price or 100.0
            offset = rng.uniform(-self.limit_offset_range, self.limit_offset_range) * mid
            price = round(mid + offset, 2)
            price = max(price, 0.01)
            order = Order(
                side=side,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=price,
                time_in_force=TimeInForce.GTC,
            )

        trades = self.book.submit(order)
        for t in trades:
            self._update_inventory(side, t.quantity, t.price)
        return [order]


class InformedTrader(BaseAgent):
    """
    Trader with a private directional signal about future price.
    Trades aggressively in the signal direction using market orders.
    Signal decays over time — informed edge diminishes as info diffuses.

    Based on the Glosten-Milgrom (1985) framework where informed traders
    know the true asset value and trade until their edge is exhausted.
    """

    def __init__(self, config: AgentConfig, book: LimitOrderBook,
                 signal_strength: float = 0.02,
                 signal_decay: float = 0.95,
                 aggression: float = 0.7):
        super().__init__(config, book)
        self.signal: float = 0.0
        self.signal_strength = signal_strength
        self.signal_decay = signal_decay
        self.aggression = aggression
        self._steps_since_signal: int = 999

    def _refresh_signal(self) -> None:
        """
        Randomly receive a new directional signal.
        Signal is a z-score: positive = bullish, negative = bearish.
        """
        if rng.random() < 0.1:  # 10% chance of new signal each step
            self.signal = rng.normal(0, self.signal_strength)
            self._steps_since_signal = 0
        else:
            self.signal *= self.signal_decay
            self._steps_since_signal += 1

    def act(self) -> list[Order]:
        self._refresh_signal()

        if abs(self.signal) < 0.001:
            return []

        side = Side.BID if self.signal > 0 else Side.ASK
        qty = max(1, int(self._sample_quantity() * self.aggression))

        if rng.random() < 0.8:
            order = Order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
            )
        else:
            mid = self.book.mid_price or 100.0
            price = round(mid + (0.05 if side == Side.BID else -0.05), 2)
            order = Order(
                side=side,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=price,
                time_in_force=TimeInForce.IOC,
            )

        trades = self.book.submit(order)
        for t in trades:
            self._update_inventory(side, t.quantity, t.price)
        return [order]