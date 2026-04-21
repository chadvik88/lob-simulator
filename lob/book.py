from collections import deque
from dataclasses import dataclass, field
from sortedcontainers import SortedDict
from typing import Optional
from .order import Order, Side, OrderType, TimeInForce
import time

@dataclass
class Trade:
    """Represents a single executed trade between two orders."""
    timestamp: float
    price: float
    quantity: int
    aggressor_order_id: int  # the incoming order that triggered the trade
    passive_order_id: int    # the resting order in the book

    def __repr__(self) -> str:
        return (
            f"Trade(price={self.price}, qty={self.quantity}, "
            f"aggressor={self.aggressor_order_id}, passive={self.passive_order_id})"
        )

@dataclass
class PriceLevel:
    """
    All orders resting at a single price point.
    Maintains FIFO queue for price-time priority.
    """
    price: float
    orders: deque = field(default_factory=deque)

    def add_order(self, order: Order) -> None:
        self.orders.append(order)

    def remove_cancelled(self) -> None:
        while self.orders and not self.orders[0].is_active:
            self.orders.popleft()

    @property
    def total_quantity(self) -> int:
        return sum(o.remaining_quantity for o in self.orders if o.is_active)

    @property
    def is_empty(self) -> bool:
        self.remove_cancelled()
        return len(self.orders) == 0

class LimitOrderBook:
    """
    A full limit order book with price-time priority matching.

    Supports limit orders, market orders, partial fills, cancellations,
    IOC and FOK time-in-force, and full trade history.

    Bids are stored in descending order (highest bid first).
    Asks are stored in ascending order (lowest ask first).
    """

    def __init__(self):
        self._bids: SortedDict = SortedDict(lambda x: -x)  # descending
        self._asks: SortedDict = SortedDict()               # ascending
        self._orders: dict[int, Order] = {}                 # order_id -> Order
        self._trades: list[Trade] = []
        self._timestamp: float = time.time()

    def submit(self, order: Order) -> list[Trade]:
        """
        Submit an order to the book.
        Returns a list of trades that resulted from this order.
        """
        self._orders[order.order_id] = order
        if order.order_type == OrderType.MARKET:
            return self._match_market(order)
        elif order.order_type == OrderType.LIMIT:
            return self._match_limit(order)
        return []

    def cancel(self, order_id: int) -> bool:
        """Cancel an order by ID. Returns True if successful."""
        order = self._orders.get(order_id)
        if order is None:
            return False
        success = order.cancel()
        if success:
            self._clean_price_level(order)
        return success

    def _match_market(self, order: Order) -> list[Trade]:
        """
        Match a market order against the best available prices.
        Consumes liquidity until filled or book is exhausted.
        IOC/FOK logic applied after matching.
        """
        trades = []
        book_side = self._asks if order.side == Side.BID else self._bids

        if order.time_in_force == TimeInForce.FOK:
            available = sum(
                level.total_quantity
                for level in book_side.values()
            )
            if available < order.quantity:
                order.cancel()
                return []

        while not order.is_filled and book_side:
            best_price = next(iter(book_side))
            level = book_side[best_price]
            level.remove_cancelled()

            while not order.is_filled and level.orders:
                passive = level.orders[0]
                if not passive.is_active:
                    level.orders.popleft()
                    continue

                fill_qty = min(order.remaining_quantity, passive.remaining_quantity)
                order.fill(fill_qty)
                passive.fill(fill_qty)

                trade = Trade(
                    timestamp=time.time(),
                    price=best_price,
                    quantity=fill_qty,
                    aggressor_order_id=order.order_id,
                    passive_order_id=passive.order_id,
                )
                trades.append(trade)
                self._trades.append(trade)

                if passive.is_filled:
                    level.orders.popleft()

            if level.is_empty:
                del book_side[best_price]

        return trades

    def _match_limit(self, order: Order) -> list[Trade]:
        """
        Match a limit order. First tries to match against the book,
        then rests any unfilled quantity (unless IOC/FOK).
        """
        trades = []
        opposite = self._asks if order.side == Side.BID else self._bids

        if order.time_in_force == TimeInForce.FOK:
            available = sum(
                level.total_quantity
                for price, level in opposite.items()
                if (order.side == Side.BID and price <= order.price) or
                   (order.side == Side.ASK and price >= order.price)
            )
            if available < order.quantity:
                order.cancel()
                return []

        while not order.is_filled and opposite:
            best_price = next(iter(opposite))

            crosses = (
                (order.side == Side.BID and best_price <= order.price) or
                (order.side == Side.ASK and best_price >= order.price)
            )
            if not crosses:
                break

            level = opposite[best_price]
            level.remove_cancelled()

            while not order.is_filled and level.orders:
                passive = level.orders[0]
                if not passive.is_active:
                    level.orders.popleft()
                    continue

                fill_qty = min(order.remaining_quantity, passive.remaining_quantity)
                order.fill(fill_qty)
                passive.fill(fill_qty)

                trade = Trade(
                    timestamp=time.time(),
                    price=best_price,
                    quantity=fill_qty,
                    aggressor_order_id=order.order_id,
                    passive_order_id=passive.order_id,
                )
                trades.append(trade)
                self._trades.append(trade)

                if passive.is_filled:
                    level.orders.popleft()

            if level.is_empty:
                del opposite[best_price]

        if not order.is_filled and not order.is_cancelled:
            if order.time_in_force == TimeInForce.IOC:
                order.cancel()
            else:
                self._rest_order(order)

        return trades

    def _rest_order(self, order: Order) -> None:
        """Place an unfilled limit order into the book."""
        book_side = self._bids if order.side == Side.BID else self._asks
        if order.price not in book_side:
            book_side[order.price] = PriceLevel(price=order.price)
        book_side[order.price].add_order(order)

    def _clean_price_level(self, order: Order) -> None:
        """Remove empty price levels after a cancellation."""
        book_side = self._bids if order.side == Side.BID else self._asks
        if order.price in book_side and book_side[order.price].is_empty:
            del book_side[order.price]

    @property
    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return next(iter(self._bids))

    @property
    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return next(iter(self._asks))

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2

    def depth_snapshot(self, levels: int = 10) -> dict:
        """
        Returns the top N price levels on each side.
        Format: { 'bids': [(price, qty), ...], 'asks': [(price, qty), ...] }
        """
        bids = [
            (price, level.total_quantity)
            for price, level in list(self._bids.items())[:levels]
            if not level.is_empty
        ]
        asks = [
            (price, level.total_quantity)
            for price, level in list(self._asks.items())[:levels]
            if not level.is_empty
        ]
        return {"bids": bids, "asks": asks}

    @property
    def trade_history(self) -> list[Trade]:
        return list(self._trades)

    @property
    def order_count(self) -> int:
        return sum(1 for o in self._orders.values() if o.is_active)

    def __repr__(self) -> str:
        return (
            f"LimitOrderBook(best_bid={self.best_bid}, best_ask={self.best_ask}, "
            f"spread={self.spread}, orders={self.order_count})"
        )