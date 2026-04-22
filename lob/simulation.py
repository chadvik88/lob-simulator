import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import threading
import time as real_time
from .book import LimitOrderBook, Trade
from .agents import BaseAgent, MarketMaker, NoiseTrader, InformedTrader, AgentConfig

rng = np.random.default_rng()

@dataclass
class SimulationConfig:
    duration: float = 3600.0
    n_market_makers: int = 3
    n_noise_traders: int = 10
    n_informed_traders: int = 2
    mm_arrival_rate: float = 5.0
    noise_arrival_rate: float = 3.0
    informed_arrival_rate: float = 1.0
    mm_spread: float = 0.10
    mm_max_inventory: int = 500
    mm_skew_factor: float = 0.01
    noise_market_order_prob: float = 0.3
    informed_signal_strength: float = 0.02
    informed_signal_decay: float = 0.95
    informed_aggression: float = 0.7
    snapshot_interval: float = 1.0


@dataclass
class SimulationSnapshot:
    time: float
    mid_price: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    bid_depth: int
    ask_depth: int
    trade_count: int
    depth_snapshot: dict = field(default_factory=dict)


class StreamingBuffer:
    """
    Thread-safe buffer that streams snapshots and trades
    from the simulation to the dashboard in real time.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.snapshots: list[SimulationSnapshot] = []
        self.trades: list[Trade] = []
        self.agent_states: list[dict] = []
        self.running: bool = False
        self.progress: float = 0.0
        self.done: bool = False

    def push_snapshot(self, snapshot: SimulationSnapshot) -> None:
        with self._lock:
            self.snapshots.append(snapshot)

    def push_trade(self, trade: Trade) -> None:
        with self._lock:
            self.trades.append(trade)

    def push_agent_state(self, state: dict) -> None:
        with self._lock:
            self.agent_states.append(state)

    def read(self) -> tuple:
        with self._lock:
            return (
                list(self.snapshots),
                list(self.trades),
                list(self.agent_states),
            )

    def reset(self) -> None:
        with self._lock:
            self.snapshots.clear()
            self.trades.clear()
            self.agent_states.clear()
            self.running = False
            self.progress = 0.0
            self.done = False


@dataclass
class SimulationResult:
    snapshots: list[SimulationSnapshot]
    trades: list[Trade]
    agents: list[BaseAgent]
    book: LimitOrderBook

    @property
    def mid_prices(self) -> list[float]:
        return [s.mid_price for s in self.snapshots if s.mid_price is not None]

    @property
    def spreads(self) -> list[float]:
        return [s.spread for s in self.snapshots if s.spread is not None]

    @property
    def timestamps(self) -> list[float]:
        return [s.time for s in self.snapshots if s.mid_price is not None]

    @property
    def market_makers(self) -> list[BaseAgent]:
        return [a for a in self.agents if isinstance(a, MarketMaker)]

    @property
    def noise_traders(self) -> list[BaseAgent]:
        return [a for a in self.agents if isinstance(a, NoiseTrader)]

    @property
    def informed_traders(self) -> list[BaseAgent]:
        return [a for a in self.agents if isinstance(a, InformedTrader)]

    def agent_summary(self) -> dict:
        return {
            f"{type(a).__name__}_{a.agent_id}": {
                "pnl": round(a.pnl, 2),
                "inventory": a.inventory,
            }
            for a in self.agents
        }


class Simulation:
    def __init__(self, config: SimulationConfig = SimulationConfig(),
                 buffer: Optional[StreamingBuffer] = None):
        self.config = config
        self.book = LimitOrderBook()
        self.env = simpy.Environment()
        self.agents: list[BaseAgent] = []
        self.snapshots: list[SimulationSnapshot] = []
        self.buffer = buffer
        self._setup_agents()
        self._setup_processes()

    def _setup_agents(self) -> None:
        mm_config = AgentConfig(
            arrival_rate=self.config.mm_arrival_rate,
            min_quantity=10, max_quantity=50,
        )
        for _ in range(self.config.n_market_makers):
            self.agents.append(MarketMaker(
                config=mm_config, book=self.book,
                spread=self.config.mm_spread,
                max_inventory=self.config.mm_max_inventory,
                skew_factor=self.config.mm_skew_factor,
            ))

        noise_config = AgentConfig(
            arrival_rate=self.config.noise_arrival_rate,
            min_quantity=1, max_quantity=30,
        )
        for _ in range(self.config.n_noise_traders):
            self.agents.append(NoiseTrader(
                config=noise_config, book=self.book,
                market_order_prob=self.config.noise_market_order_prob,
            ))

        informed_config = AgentConfig(
            arrival_rate=self.config.informed_arrival_rate,
            min_quantity=10, max_quantity=80,
        )
        for _ in range(self.config.n_informed_traders):
            self.agents.append(InformedTrader(
                config=informed_config, book=self.book,
                signal_strength=self.config.informed_signal_strength,
                signal_decay=self.config.informed_signal_decay,
                aggression=self.config.informed_aggression,
            ))

    def _setup_processes(self) -> None:
        for agent in self.agents:
            self.env.process(self._agent_process(agent))
        self.env.process(self._snapshot_process())

    def _agent_process(self, agent: BaseAgent):
        while True:
            wait_time = rng.exponential(1.0 / agent.config.arrival_rate)
            yield self.env.timeout(wait_time)
            agent.act()

    def _snapshot_process(self):
        last_trade_count=0
        while True:
            yield self.env.timeout(self.config.snapshot_interval)
            real_time.sleep(0.01)
            depth = self.book.depth_snapshot(20)
            snapshot = SimulationSnapshot(
                time=self.env.now,
                mid_price=self.book.mid_price,
                best_bid=self.book.best_bid,
                best_ask=self.book.best_ask,
                spread=self.book.spread,
                bid_depth=sum(qty for _, qty in depth["bids"]),
                ask_depth=sum(qty for _, qty in depth["asks"]),
                trade_count=len(self.book.trade_history),
                depth_snapshot=depth,
            )
            self.snapshots.append(snapshot)
            if self.buffer:
                self.buffer.push_snapshot(snapshot)
                new_trades = self.book.trade_history[last_trade_count:]
                for t in new_trades:
                    self.buffer.push_trade(t)
                last_trade_count = len(self.book.trade_history)
                agent_state = {
                    "time": self.env.now,
                    "agents": {
                        f"{type(a).__name__}_{a.agent_id}": {
                            "pnl": round(a.pnl, 2),
                            "inventory": a.inventory,
                        }
                        for a in self.agents
                    }
                }
                self.buffer.push_agent_state(agent_state)
                self.buffer.progress = self.env.now / self.config.duration

    def run(self, verbose: bool = False) -> SimulationResult:
        if verbose:
            print(f"Starting simulation: {self.config.duration}s")
        if self.buffer:
            self.buffer.running = True
        self.env.run(until=self.config.duration)
        if self.buffer:
            self.buffer.running = False
            self.buffer.done = True
        return SimulationResult(
            snapshots=self.snapshots,
            trades=self.book.trade_history,
            agents=self.agents,
            book=self.book,
        )