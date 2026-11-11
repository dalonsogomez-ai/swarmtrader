"""Identidades contables con precios deterministas, sin costes ni seleccion."""
import numpy as np
import pandas as pd
import pytest

from swarmtrader.agents.genome import Genome
from swarmtrader.backtest import FeaturePanel, simulate


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_buy_and_hold_matches_asset_return_and_remaining_cash(weight):
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 100.0, 110.0, 121.0, 133.1]}, index=dates)
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "symbol"])
    features = pd.DataFrame({"signal": 1.0}, index=index)
    dollar_volume = pd.Series(1e9, index=index)
    panel = FeaturePanel(features, prices, dollar_volume)
    genome = Genome(
        weights={"signal": 1.0}, top_k=1, min_conviction=-1.0,
        rebalance_days=10**6, stop_loss=1.0, take_profit=1e9,
        min_dollar_volume=0.0, max_weight=weight,
    )

    result = simulate(panel, genome, 1, len(dates), cost_bps=0.0)

    expected_nav = 1.0 - weight + weight * prices["A"].iloc[1:].to_numpy() / 100.0
    np.testing.assert_allclose(result.equity, expected_nav, rtol=1e-12)
    assert result.total_return == pytest.approx(weight * 0.331)
