import pytest

from src.config import load_settings


def test_settings_file_parses():
    s = load_settings()
    assert s.instrument.name == "US100"
    assert s.timezones.market == "America/New_York"
    assert s.account.risk_per_trade_pct == 0.5


def test_settings_complete_and_unconfirmed_listed():
    s = load_settings()
    assert s.account.initial_balance == 2500
    assert s.missing_todos() == []
    s.require_complete()
    assert "lot_value_per_point" in s.broker.unconfirmed


def test_unfilled_todos_fail_safe():
    s = load_settings()
    s.account.daily_loss_limit_pct = None
    assert "account.daily_loss_limit_pct" in s.missing_todos()
    with pytest.raises(ValueError, match="TODO"):
        s.require_complete()
