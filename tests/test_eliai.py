"""
EliAI Trade backend test suite — covers instruments, market data, calendar,
risk, AI verdict, AI coach, trade journal, equity curve.
"""
import os
import time
import pytest
import requests
from urllib.parse import quote

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    # Fallback: read from frontend env
    with open("/app/frontend/.env") as f:
        for line in f:
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = line.split("=", 1)[1].strip().rstrip("/")

API = f"{BASE_URL}/api"

EXPECTED_SYMBOLS = {"AUD/USD", "USD/JPY", "EUR/USD", "ASX 200", "NAS100", "S&P 500", "FTSE 100", "XAU/USD"}
ALLOWED_CCYS = {"USD", "EUR", "GBP", "JPY", "AUD"}


@pytest.fixture(scope="module")
def s():
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    return sess


# ===== Instruments =====
class TestInstruments:
    def test_get_instruments(self, s):
        r = s.get(f"{API}/instruments", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 8
        symbols = {i["symbol"] for i in data["instruments"]}
        assert symbols == EXPECTED_SYMBOLS
        # No crypto leftover
        for inst in data["instruments"]:
            assert inst["asset_class"] in {"forex", "indices", "commodities"}


# ===== Market =====
class TestMarket:
    def test_market_all(self, s):
        r = s.get(f"{API}/market/all", timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        symbols = {i["symbol"] for i in data["instruments"]}
        # Yahoo can rate-limit; assert at least 5 of 8 returned and all are within expected
        assert len(symbols) >= 5
        assert symbols.issubset(EXPECTED_SYMBOLS)
        for inst in data["instruments"]:
            assert "price" in inst and inst["price"] is not None

    def test_historical_eur_usd_7d(self, s):
        sym = quote("EUR/USD", safe="")
        r = s.get(f"{API}/market/{sym}/historical?range=7d", timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["symbol"] == "EUR/USD"
        assert len(data["candles"]) > 0

    def test_historical_xau_usd_1d(self, s):
        sym = quote("XAU/USD", safe="")
        r = s.get(f"{API}/market/{sym}/historical?range=1d", timeout=30)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["symbol"] == "XAU/USD"


# ===== Sessions =====
class TestSessions:
    def test_sessions_status(self, s):
        r = s.get(f"{API}/sessions/status", timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert "active_overlaps" in data
        assert set(data["sessions"].keys()) == {"sydney", "tokyo", "london", "new_york"}


# ===== Calendar =====
class TestCalendar:
    def test_calendar_filtering(self, s):
        r = s.get(f"{API}/calendar/events", timeout=30)
        assert r.status_code == 200
        data = r.json()
        # Should be 2 or 3 star only and currencies restricted
        for e in data["events"]:
            assert e["stars"] in (2, 3), f"got stars={e['stars']}"
            assert e["currency"] in ALLOWED_CCYS, f"bad currency {e['currency']}"


# ===== Risk Calculator =====
class TestRisk:
    def test_risk_calculate(self, s):
        payload = {
            "account_balance": 10000,
            "risk_percentage": 1,
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "symbol": "EUR/USD",
        }
        r = s.post(f"{API}/risk/calculate", json=payload, timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["risk_amount"] == 100.0
        assert d["pip_risk"] == 0.005
        assert d["position_size"] > 0
        assert d["potential_profit_2r"] == 200.0


# ===== AI Engine =====
class TestAI:
    def test_get_rules(self, s):
        r = s.get(f"{API}/ai/rules", timeout=15)
        assert r.status_code == 200
        assert "Smart Money" in r.json()["rules"] or "SMC" in r.json()["rules"] or len(r.json()["rules"]) > 100

    def test_update_rules_roundtrip(self, s):
        original = s.get(f"{API}/ai/rules", timeout=15).json()["rules"]
        new_rules = original + "\n\n# TEST_MARKER 42"
        r = s.put(f"{API}/ai/rules", json={"rules": new_rules}, timeout=15)
        assert r.status_code == 200
        check = s.get(f"{API}/ai/rules", timeout=15).json()["rules"]
        assert "TEST_MARKER 42" in check
        # Restore
        s.put(f"{API}/ai/rules", json={"rules": original}, timeout=15)

    def test_ai_verdict_eur_usd(self, s):
        sym = quote("EUR/USD", safe="")
        r = s.get(f"{API}/ai/verdict/{sym}", timeout=90)
        assert r.status_code == 200, r.text
        data = r.json()
        v = data["verdict"]
        assert v["action"] in {"BUY", "SELL", "HOLD", "WATCH"}
        assert 0 <= v["confidence"] <= 100
        assert v["bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}
        assert "entry" in v["key_levels"]
        assert "stop_loss" in v["key_levels"]
        assert "take_profit" in v["key_levels"]
        assert isinstance(v.get("reasoning"), str) and len(v["reasoning"]) > 0

    def test_ai_verdicts_cached(self, s):
        r = s.get(f"{API}/ai/verdicts", timeout=15)
        assert r.status_code == 200
        data = r.json()
        # Should at least include EUR/USD now after previous test
        assert data["count"] >= 1


# ===== Trade Journal =====
class TestTrades:
    trade_id = None

    def test_create_trade(self, s):
        payload = {
            "instrument": "EUR/USD",
            "direction": "LONG",
            "session": "LONDON",
            "entry_model": "FVG + Liquidity Sweep",
            "entry_zone": "Premium OB on H1",
            "entry_price": 1.1000,
            "stop_loss": 1.0950,
            "take_profit": 1.1100,
            "position_size": 10000,
            "notes": "TEST_trade",
        }
        r = s.post(f"{API}/trades", json=payload, timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["instrument"] == "EUR/USD"
        assert d["status"] == "OPEN"
        TestTrades.trade_id = d["id"]

    def test_list_and_stats(self, s):
        r = s.get(f"{API}/trades", timeout=15)
        assert r.status_code == 200
        assert any(t["id"] == TestTrades.trade_id for t in r.json())
        r2 = s.get(f"{API}/trades/stats", timeout=15)
        assert r2.status_code == 200
        assert r2.json()["total_trades"] >= 1

    def test_close_trade_long_pnl(self, s):
        tid = TestTrades.trade_id
        assert tid
        r = s.put(f"{API}/trades/{tid}",
                  json={"status": "CLOSED", "exit_price": 1.1050}, timeout=15)
        assert r.status_code == 200
        # Verify P/L: (1.1050-1.1000)*10000 = 50
        listing = s.get(f"{API}/trades", timeout=15).json()
        t = next(x for x in listing if x["id"] == tid)
        assert t["status"] == "CLOSED"
        assert abs(t["profit_loss"] - 50.0) < 0.01, f"got {t['profit_loss']}"

    def test_short_trade_pnl(self, s):
        payload = {
            "instrument": "USD/JPY", "direction": "SHORT", "session": "NEW_YORK",
            "entry_model": "OB rejection", "entry_price": 150.00,
            "stop_loss": 150.50, "take_profit": 149.00, "position_size": 1000,
        }
        r = s.post(f"{API}/trades", json=payload, timeout=15)
        assert r.status_code == 200
        tid = r.json()["id"]
        r2 = s.put(f"{API}/trades/{tid}", json={"status": "CLOSED", "exit_price": 149.50}, timeout=15)
        assert r2.status_code == 200
        listing = s.get(f"{API}/trades", timeout=15).json()
        t = next(x for x in listing if x["id"] == tid)
        # SHORT: (150 - 149.5)*1000 = 500
        assert abs(t["profit_loss"] - 500.0) < 0.01, f"got {t['profit_loss']}"
        TestTrades.short_id = tid

    def test_ai_coach_score(self, s):
        tid = TestTrades.trade_id
        r = s.post(f"{API}/ai/coach/score/{tid}", timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        scores = d["ai_coach"]["scores"]
        for k in ("setup_quality", "entry_execution", "risk_management", "patience_discipline"):
            assert 0 <= scores[k] <= 10, f"{k}={scores[k]}"
        assert 0 <= d["ai_coach"]["overall_score"] <= 10
        for f in ("what_you_did_well", "areas_to_improve", "key_lesson"):
            assert f in d["ai_coach"]

    def test_coach_patterns(self, s):
        r = s.get(f"{API}/ai/coach/patterns", timeout=90)
        assert r.status_code == 200
        assert "patterns" in r.json()

    def test_equity_curve(self, s):
        r = s.get(f"{API}/reports/equity-curve", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert "curve" in d
        assert d["trades"] >= 2

    def test_delete_trade(self, s):
        tid = TestTrades.trade_id
        r = s.delete(f"{API}/trades/{tid}", timeout=15)
        assert r.status_code == 200
        # cleanup short trade too
        if hasattr(TestTrades, "short_id"):
            s.delete(f"{API}/trades/{TestTrades.short_id}", timeout=15)
