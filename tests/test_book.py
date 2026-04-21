import pytest
from lob.order import Order, Side, OrderType, TimeInForce
from lob.book import LimitOrderBook, Trade

def make_limit(side, price, qty, tif=TimeInForce.GTC):
    return Order(side=side, order_type=OrderType.LIMIT, quantity=qty, price=price, time_in_force=tif)

def make_market(side, qty):
    return Order(side=side, order_type=OrderType.MARKET, quantity=qty, time_in_force=TimeInForce.IOC)

def test_basic_limit_match():
    book = LimitOrderBook()
    sell = make_limit(Side.ASK, price=100.0, qty=10)
    buy = make_limit(Side.BID, price=100.0, qty=10)
    book.submit(sell)
    trades = book.submit(buy)
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].quantity == 10
    assert buy.is_filled
    assert sell.is_filled

def test_partial_fill():
    book = LimitOrderBook()
    sell = make_limit(Side.ASK, price=100.0, qty=10)
    buy = make_limit(Side.BID, price=100.0, qty=6)
    book.submit(sell)
    trades = book.submit(buy)
    assert len(trades) == 1
    assert trades[0].quantity == 6
    assert buy.is_filled
    assert sell.remaining_quantity == 4
    assert not sell.is_filled

def test_partial_fill_opposite():
    book = LimitOrderBook()
    sell = make_limit(Side.ASK, price=100.0, qty=5)
    buy = make_limit(Side.BID, price=100.0, qty=10)
    book.submit(sell)
    trades = book.submit(buy)
    assert trades[0].quantity == 5
    assert sell.is_filled
    assert buy.remaining_quantity == 5
    snapshot = book.depth_snapshot()
    assert snapshot["bids"][0][0] == 100.0
    assert snapshot["bids"][0][1] == 5

def test_no_cross():
    book = LimitOrderBook()
    buy = make_limit(Side.BID, price=99.0, qty=10)
    sell = make_limit(Side.ASK, price=101.0, qty=10)
    book.submit(buy)
    book.submit(sell)
    assert len(book.trade_history) == 0
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0
    assert book.spread == 2.0

def test_market_order_full_fill():
    book = LimitOrderBook()
    book.submit(make_limit(Side.ASK, price=100.0, qty=10))
    buy = make_market(Side.BID, qty=10)
    trades = book.submit(buy)
    assert buy.is_filled
    assert len(trades) == 1

def test_market_order_empty_book():
    book = LimitOrderBook()
    buy = make_market(Side.BID, qty=10)
    trades = book.submit(buy)
    assert len(trades) == 0
    assert not buy.is_filled

def test_market_order_sweeps_multiple_levels():
    book = LimitOrderBook()
    book.submit(make_limit(Side.ASK, price=100.0, qty=5))
    book.submit(make_limit(Side.ASK, price=101.0, qty=5))
    book.submit(make_limit(Side.ASK, price=102.0, qty=5))
    buy = make_market(Side.BID, qty=12)
    trades = book.submit(buy)
    total_filled = sum(t.quantity for t in trades)
    assert total_filled == 12
    assert buy.is_filled

def test_price_time_priority():
    book = LimitOrderBook()
    first = make_limit(Side.ASK, price=100.0, qty=5)
    second = make_limit(Side.ASK, price=100.0, qty=5)
    book.submit(first)
    book.submit(second)
    buy = make_market(Side.BID, qty=5)
    trades = book.submit(buy)
    assert trades[0].passive_order_id == first.order_id

def test_cancellation():
    book = LimitOrderBook()
    sell = make_limit(Side.ASK, price=100.0, qty=10)
    book.submit(sell)
    assert book.cancel(sell.order_id) == True
    assert sell.is_cancelled
    snapshot = book.depth_snapshot()
    assert len(snapshot["asks"]) == 0

def test_cancel_filled_order():
    book = LimitOrderBook()
    sell = make_limit(Side.ASK, price=100.0, qty=10)
    buy = make_limit(Side.BID, price=100.0, qty=10)
    book.submit(sell)
    book.submit(buy)
    assert book.cancel(sell.order_id) == False

def test_cancel_nonexistent_order():
    book = LimitOrderBook()
    assert book.cancel(99999) == False

def test_ioc_partial():
    book = LimitOrderBook()
    book.submit(make_limit(Side.ASK, price=100.0, qty=5))
    buy = Order(side=Side.BID, order_type=OrderType.LIMIT,
                quantity=10, price=100.0, time_in_force=TimeInForce.IOC)
    trades = book.submit(buy)
    assert trades[0].quantity == 5
    assert buy.is_cancelled
    assert buy.filled_quantity == 5

def test_fok_success():
    book = LimitOrderBook()
    book.submit(make_limit(Side.ASK, price=100.0, qty=10))
    buy = Order(side=Side.BID, order_type=OrderType.LIMIT,
                quantity=10, price=100.0, time_in_force=TimeInForce.FOK)
    trades = book.submit(buy)
    assert buy.is_filled
    assert len(trades) == 1

def test_fok_fail():
    book = LimitOrderBook()
    book.submit(make_limit(Side.ASK, price=100.0, qty=5))
    buy = Order(side=Side.BID, order_type=OrderType.LIMIT,
                quantity=10, price=100.0, time_in_force=TimeInForce.FOK)
    trades = book.submit(buy)
    assert len(trades) == 0
    assert buy.is_cancelled
    snapshot = book.depth_snapshot()
    assert snapshot["asks"][0][1] == 5

def test_mid_price_and_spread():
    book = LimitOrderBook()
    book.submit(make_limit(Side.BID, price=99.0, qty=10))
    book.submit(make_limit(Side.ASK, price=101.0, qty=10))
    assert book.mid_price == 100.0
    assert book.spread == 2.0

def test_depth_snapshot_ordering():
    book = LimitOrderBook()
    for price in [99, 98, 97]:
        book.submit(make_limit(Side.BID, price=float(price), qty=10))
    for price in [101, 102, 103]:
        book.submit(make_limit(Side.ASK, price=float(price), qty=10))
    snapshot = book.depth_snapshot()
    bid_prices = [p for p, _ in snapshot["bids"]]
    ask_prices = [p for p, _ in snapshot["asks"]]
    assert bid_prices == sorted(bid_prices, reverse=True)
    assert ask_prices == sorted(ask_prices)