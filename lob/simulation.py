import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .book import LimitOrderBook, Trade
from .agents import BaseAgent, MarketMaker, NoiseTrader, InformedTrader, AgentConfig

rng = np.random.default_rng()

@dataclass
class SimulationConfig:
    """
    Full configuration for a simulation run.
    All time units are in seconds of simulated time.
    """
    duration: float = 3600.0          # 1 hour of simulated time
    n_market_makers: int = 3
    n_noise_traders: int = 10
    n_informed_traders: int = 2
    mm_arrival_rate: float = 5.0      # requotes per second
    noise_arrival_rate: float = 3.0   # orders per second
    informed_arrival_rate: float = 1.0
    mm_spread: float = 0.10
    mm_max_inventory: int = 500
    mm_skew_factor: float = 0.01
    noise_market_order_prob: float = 0.3
    informed_signal_strength: float = 0.02
    informed_signal_decay: float = 0.95
    informed_aggression: float = 0.7
    snapshot_interval: float = 1.0    # how often to record book state


@dataclass
class SimulationSnapshot:
    """Point-in-time record of book state during simulation."""
    time: float
    mid_price: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    spread: Optional[float]
    bid_depth: int
    ask_depth: int
    trade_count: int


class Simulation:
    """
    Discrete-event simulation of a limit order book market.
    Uses SimPy to manage concurrent agent processes.

    Each agent runs as an independent SimPy process, sleeping
    for exponentially distributed inter-arrival times (Poisson process)
    before acting on the shared order book.
    """

    def __init__(self, config: SimulationConfig = SimulationConfig()):
        self.config = config
        self.book = LimitOrderBook()
        self.env = simpy.Environment()
        self.agents: list[BaseAgent] = []
        self.snapshots: list[SimulationSnapshot] = []
        self._trade_count_at_last_snapshot: int = 0
        self._setup_agents()
        self._setup_processes()

    def _setup_agents(self) -> None:
        mm_config = AgentConfig(
            arrival_rate=self.config.mm_arrival_rate,
            min_quantity=10,
            max_quantity=50,
        )
        for _ in range(self.config.n_market_makers):
            self.agents.append(MarketMaker(
                config=mm_config,
                book=self.book,
                spread=self.config.mm_spread,
                max_inventory=self.config.mm_max_inventory,
                skew_factor=self.config.mm_skew_factor,
            ))

        noise_config = AgentConfig(
            arrival_rate=self.config.noise_arrival_rate,
            min_quantity=1,
            max_quantity=30,
        )
        for _ in range(self.config.n_noise_traders):
            self.agents.append(NoiseTrader(
                config=noise_config,
                book=self.book,
                market_order_prob=self.config.noise_market_order_prob,
            ))

        informed_config = AgentConfig(
            arrival_rate=self.config.informed_arrival_rate,
            min_quantity=10,
            max_quantity=80,
        )
        for _ in range(self.config.n_informed_traders):
            self.agents.append(InformedTrader(
                config=informed_config,
                book=self.book,
                signal_strength=self.config.informed_signal_strength,
                signal_decay=self.config.informed_signal_decay,
                aggression=self.config.informed_aggression,
            ))

    def _setup_processes(self) -> None:
        for agent in self.agents:
            self.env.process(self._agent_process(agent))
        self.env.process(self._snapshot_process())

    def _agent_process(self, agent: BaseAgent):
        """
        SimPy process for a single agent.
        Agent sleeps for a Poisson-distributed interval then acts.
        Runs for the full simulation duration.
        """
        while True:
            wait_time = rng.exponential(1.0 / agent.config.arrival_rate)
            yield self.env.timeout(wait_time)
            agent.act()

    def _snapshot_process(self):
        """
        Records book state at regular intervals for analytics.
        """
        while True:
            yield self.env.timeout(self.config.snapshot_interval)
            trades = self.book.trade_history
            snapshot = SimulationSnapshot(
                time=self.env.now,
                mid_price=self.book.mid_price,
                best_bid=self.book.best_bid,
                best_ask=self.book.best_ask,
                spread=self.book.spread,
                bid_depth=sum(qty for _, qty in self.book.depth_snapshot(20)["bids"]),
                ask_depth=sum(qty for _, qty in self.book.depth_snapshot(20)["asks"]),
                trade_count=len(trades),
            )
            self.snapshots.append(snapshot)

    def run(self, verbose: bool = False) -> "SimulationResult":
        """
        Run the full simulation.
        Returns a SimulationResult with all snapshots and trade history.
        """
        if verbose:
            print(f"Starting simulation: {self.config.duration}s simulated time")
            print(f"Agents: {self.config.n_market_makers} MMs, "
                  f"{self.config.n_noise_traders} noise, "
                  f"{self.config.n_informed_traders} informed")

        self.env.run(until=self.config.duration)

        result = SimulationResult(
            snapshots=self.snapshots,
            trades=self.book.trade_history,
            agents=self.agents,
            book=self.book,
        )

        if verbose:
            print(f"Simulation complete.")
            print(f"Total trades: {len(result.trades)}")
            print(f"Final mid price: {self.book.mid_price}")
            print(f"Final spread: {self.book.spread}")

        return result


@dataclass
class SimulationResult:
    """
    Complete output of a simulation run.
    Contains all snapshots, trades, agent states, and final book.
    """
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
        """Returns PnL and inventory summary for all agents."""
        return {
            f"{type(a).__name__}_{a.agent_id}": {
                "pnl": round(a.pnl, 2),
                "inventory": a.inventory,
            }
            for a in self.agents
        }