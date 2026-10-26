"""Central configuration. This project needs no credentials of any kind."""
from dataclasses import dataclass

DATA_SOURCE_NAME = "Delta Exchange India public REST API (api.india.delta.exchange)"
BASE_URL = "https://api.india.delta.exchange"
CANDLES_PATH = "/v2/history/candles"

# Delta India lists BTC and ETH as USD-quoted perpetual futures (not BTC/USDT spot).
SYMBOLS = {"BTC": "BTCUSD", "ETH": "ETHUSD"}
TF_SECONDS = {"5m": 300, "15m": 900}

# PLACEHOLDER cost assumptions per side. Verify against Delta's current fee schedule
# (including taxes) and edit here or in the dashboard sidebar.
DEFAULT_FEE_RATE = 0.0005        # 0.05% taker fee per side
DEFAULT_SLIPPAGE_RATE = 0.0003   # 0.03% adverse slippage on market-style fills


@dataclass(frozen=True)
class Params:
    # indicators
    rsi_len: int = 14
    atr_len: int = 14
    vol_len: int = 20
    er_len: int = 20
    # structure
    swing_left: int = 3
    swing_right: int = 3
    structure_lookback: int = 120
    min_bars_5m: int = 150
    min_bars_15m: int = 100
    # bias / scoring
    bias_threshold: int = 3
    min_score: int = 6
    # regime thresholds
    er_trend: float = 0.30
    er_range: float = 0.18
    vol_high_ratio: float = 2.0
    low_vol_rel: float = 0.6
    # trade construction
    atr_stop_buffer: float = 0.5
    min_risk_atr: float = 0.5
    max_risk_atr: float = 3.0
    min_room_r: float = 1.0
    min_net_rr: float = 1.5
    max_extension_atr: float = 1.5
    # execution model (backtest + paper)
    target_r: float = 2.0
    max_hold_bars: int = 48
    max_chase_atr: float = 0.5
    fee_rate: float = DEFAULT_FEE_RATE
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE
