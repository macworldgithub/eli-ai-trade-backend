"""
Iteration 3 tests:
 - Alerts module (config, recipients CRUD, preview, test-send, dispatch-now, history)
 - Editable strategy rules persistence
 - Regression: trade journal/stats still work; trades not wiped on startup
"""
import os
import re
import uuid
import pytest
import requests
from urllib.parse import quote

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
if not BASE_URL:
    with open("/app/frontend/.env") as f:
        for line in f:
            if line.startswith("REACT_APP_BACKEND_URL="):
                BASE_URL = line.split("=", 1)[1].strip().rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def s():
    sess = requests.Session()
    sess.headers.update({"Content-Type": "application/json"})
    return sess


# ===== Alerts: config =====
class TestAlertsConfig:
    def test_alerts_config(self, s):
        r = s.get(f"{API}/alerts/config", timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        # SMTP not configured in this env
        assert d["smtp_configured"] is False
        assert d["lead_minutes"] == 30
        assert d["scheduler_running"] is True
        assert "setup_hint" in d


# ===== Alerts: recipients CRUD =====
class TestRecipients:
    def test_seeded_recipients_present(self, s):
        r = s.get(f"{API}/alerts/recipients", timeout=15)
        assert r.status_code == 200
        d = r.json()
        emails = {x["email"] for x in d["recipients"]}
        assert {"brodie@eliai.trade", "mark@eliai.trade", "brooklyn@eliai.trade"}.issubset(emails)
        for x in d["recipients"]:
            if x["email"] in {"brodie@eliai.trade", "mark@eliai.trade", "brooklyn@eliai.trade"}:
                assert x["enabled"] is True

    def test_add_recipient_valid(self, s):
        unique = f"test_user_{uuid.uuid4().hex[:8]}@eliai.trade"
        r = s.post(f"{API}/alerts/recipients", json={"email": unique}, timeout=15)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["email"] == unique
        assert d["enabled"] is True
        assert "id" in d
        # GET verifies persistence
        listing = s.get(f"{API}/alerts/recipients", timeout=15).json()["recipients"]
        assert any(x["email"] == unique for x in listing)
        # cleanup
        s.delete(f"{API}/alerts/recipients/{d['id']}", timeout=15)

    def test_add_recipient_duplicate(self, s):
        # duplicate brodie should 409
        r = s.post(f"{API}/alerts/recipients", json={"email": "brodie@eliai.trade"}, timeout=15)
        assert r.status_code == 409

    def test_add_recipient_invalid(self, s):
        r = s.post(f"{API}/alerts/recipients", json={"email": "not-an-email"}, timeout=15)
        assert r.status_code == 400

    def test_toggle_recipient(self, s):
        unique = f"TEST_toggle_{uuid.uuid4().hex[:8]}@eliai.trade"
        created = s.post(f"{API}/alerts/recipients", json={"email": unique}, timeout=15).json()
        rid = created["id"]
        r = s.put(f"{API}/alerts/recipients/{rid}",
                  json={"email": unique, "enabled": False}, timeout=15)
        assert r.status_code == 200
        assert r.json()["enabled"] is False
        # verify persisted
        listing = s.get(f"{API}/alerts/recipients", timeout=15).json()["recipients"]
        match = next(x for x in listing if x["id"] == rid)
        assert match["enabled"] is False
        # cleanup
        s.delete(f"{API}/alerts/recipients/{rid}", timeout=15)

    def test_delete_recipient(self, s):
        unique = f"TEST_del_{uuid.uuid4().hex[:8]}@eliai.trade"
        created = s.post(f"{API}/alerts/recipients", json={"email": unique}, timeout=15).json()
        rid = created["id"]
        r = s.delete(f"{API}/alerts/recipients/{rid}", timeout=15)
        assert r.status_code == 200
        # 404 for unknown id
        r2 = s.delete(f"{API}/alerts/recipients/non-existent-id-xyz", timeout=15)
        assert r2.status_code == 404


# ===== Alerts: preview / test-send / dispatch / history =====
class TestAlertOps:
    def test_preview(self, s):
        r = s.post(f"{API}/alerts/preview", timeout=30)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "event" in d and "subject" in d and "html" in d
        subj = d["subject"]
        # subject should contain stars and currency and title fragment
        assert "★" in subj
        assert any(c in subj for c in ["USD", "EUR", "GBP", "JPY", "AUD"])
        assert len(d["html"]) > 200

    def test_test_send_simulated(self, s):
        r = s.post(f"{API}/alerts/test-send", timeout=30)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["sent"] is False
        assert d["reason"] == "SMTP not configured"
        assert isinstance(d["recipients"], list)
        assert len(d["recipients"]) >= 1

    def test_dispatch_now(self, s):
        r = s.post(f"{API}/alerts/dispatch-now", timeout=45)
        assert r.status_code == 200, r.text
        d = r.json()
        for k in ("dispatched", "skipped", "checked_events", "recipients", "smtp_configured"):
            assert k in d
        assert d["smtp_configured"] is False

    def test_history(self, s):
        r = s.get(f"{API}/alerts/history", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert "history" in d
        assert isinstance(d["history"], list)


# ===== Editable strategy rules =====
class TestRules:
    def test_default_rules_present(self, s):
        r = s.get(f"{API}/ai/rules", timeout=15)
        assert r.status_code == 200
        rules = r.json()["rules"]
        # default SMC framework should be ~1444 chars
        assert len(rules) > 800
        assert "Smart Money" in rules or "SMC" in rules or "MARKET STRUCTURE" in rules

    def test_rules_roundtrip_persistence(self, s):
        original = s.get(f"{API}/ai/rules", timeout=15).json()["rules"]
        marker = f"TEST_MARKER_{uuid.uuid4().hex[:6]}"
        new_rules = original + f"\n\n# {marker}"
        r = s.put(f"{API}/ai/rules", json={"rules": new_rules}, timeout=15)
        assert r.status_code == 200
        # GET to verify persistence
        check = s.get(f"{API}/ai/rules", timeout=15).json()["rules"]
        assert marker in check
        # Restore original
        s.put(f"{API}/ai/rules", json={"rules": original}, timeout=15)
        restored = s.get(f"{API}/ai/rules", timeout=15).json()["rules"]
        assert marker not in restored


# ===== Regression: trades & stats =====
class TestTradesRegression:
    def test_trade_lifecycle_and_stats(self, s):
        before = s.get(f"{API}/trades", timeout=15).json()
        before_count = len(before)
        payload = {
            "instrument": "EUR/USD", "direction": "LONG", "session": "LONDON",
            "entry_model": "FVG", "entry_price": 1.1, "stop_loss": 1.095,
            "take_profit": 1.11, "position_size": 1000, "notes": "TEST_regression",
        }
        r = s.post(f"{API}/trades", json=payload, timeout=15)
        assert r.status_code == 200
        tid = r.json()["id"]
        # Stats still works (regression of null pl handling)
        st = s.get(f"{API}/trades/stats", timeout=15)
        assert st.status_code == 200
        assert st.json()["total_trades"] >= before_count + 1
        # cleanup
        s.delete(f"{API}/trades/{tid}", timeout=15)

    def test_ai_verdict_still_works(self, s):
        sym = quote("EUR/USD", safe="")
        r = s.get(f"{API}/ai/verdict/{sym}", timeout=90)
        assert r.status_code == 200, r.text
        v = r.json()["verdict"]
        assert v["action"] in {"BUY", "SELL", "HOLD", "WATCH"}
        assert v["bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}


# ===== Startup safety: ensure no delete_many on trades =====
class TestStartupSafety:
    def test_no_trades_wipe_in_server(self):
        with open("/app/backend/server.py") as f:
            src = f.read()
        # Should NOT contain a destructive wipe on startup
        assert not re.search(r"db\.trades\.delete_many", src), \
            "server.py contains db.trades.delete_many — trades would be wiped on restart!"
