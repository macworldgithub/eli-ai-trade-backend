"""
EliAI Trade — Backend API
AI-powered trading platform for professional traders.
8 instruments: AUD/USD, USD/JPY, EUR/USD, ASX 200, NAS100, S&P 500, FTSE 100, XAU/USD
"""
from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import json
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import smtplib
from email.message import EmailMessage
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

app = FastAPI(title="EliAI Trade API", version="2.0.0")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============== INSTRUMENT REGISTRY (THE 8) ==============

INSTRUMENTS = [
    # Forex
    {"symbol": "AUD/USD", "name": "Australian Dollar / US Dollar", "asset_class": "forex", "yahoo": "AUDUSD=X", "currencies": ["AUD", "USD"]},
    {"symbol": "USD/JPY", "name": "US Dollar / Japanese Yen", "asset_class": "forex", "yahoo": "USDJPY=X", "currencies": ["USD", "JPY"]},
    {"symbol": "EUR/USD", "name": "Euro / US Dollar", "asset_class": "forex", "yahoo": "EURUSD=X", "currencies": ["EUR", "USD"]},
    # Indices
    {"symbol": "ASX 200", "name": "Australian Securities Exchange 200", "asset_class": "indices", "yahoo": "^AXJO", "currencies": ["AUD"]},
    {"symbol": "NAS100", "name": "Nasdaq 100", "asset_class": "indices", "yahoo": "^NDX", "currencies": ["USD"]},
    {"symbol": "S&P 500", "name": "S&P 500", "asset_class": "indices", "yahoo": "^GSPC", "currencies": ["USD"]},
    {"symbol": "FTSE 100", "name": "FTSE 100", "asset_class": "indices", "yahoo": "^FTSE", "currencies": ["GBP"]},
    # Commodity
    {"symbol": "XAU/USD", "name": "Gold Spot", "asset_class": "commodities", "yahoo": "GC=F", "currencies": ["USD", "XAU"]},
]

SYMBOL_LOOKUP = {i["symbol"]: i for i in INSTRUMENTS}

# Strategy rules — default SMC framework (editable via /api/ai/rules)
DEFAULT_STRATEGY_RULES = """
EliAI Trade — Strategy Framework (Smart Money Concepts)

1. MARKET STRUCTURE
   - Identify HTF (Higher Time Frame) bias: H4/D1 trend direction.
   - Look for Break of Structure (BOS) or Change of Character (CHOCH).
   - Only trade in alignment with HTF bias unless a clear CHOCH occurs.

2. LIQUIDITY
   - Mark equal highs/lows, previous day/week highs and lows (PDH/PDL).
   - Wait for liquidity sweep before entry — price grabs liquidity then reverses.

3. PREMIUM/DISCOUNT
   - Use Fibonacci 50% on the most recent leg to define premium vs discount.
   - Sell in premium, buy in discount.

4. POI (Point of Interest)
   - Fair Value Gaps (FVG), Order Blocks (OB), and Breaker Blocks are valid entry zones.
   - Inverse FVG (IFG) on the lower TF confirms entry.
   - CISD (Change in State of Delivery): confirmation of intent.
   - SMT Divergence: correlated pairs show divergence before reversal.

5. SESSION CONFLUENCE
   - Trade Asian range sweeps in London Open.
   - Trade London setups in New York Open for continuation.
   - Avoid trading during low-volume periods (e.g. Asian range mid-session).

6. RISK MANAGEMENT
   - Max 5% risk per trade. Aim for 1:2 RR minimum.
   - 3–5 trades per week.
   - Stop loss above/below the swept liquidity or POI invalidation.

7. NEWS FILTER
   - Avoid entries 15 minutes before / after High (3-star) impact events for the traded currency.
   - Gold respects USD events especially CPI, NFP, FOMC.
"""

# ============== MODELS ==============

class TradeJournal(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str
    direction: str  # LONG / SHORT
    session: str    # ASIA / LONDON / NEW_YORK / OVERLAP
    entry_model: str  # e.g. "FVG + Liquidity Sweep"
    entry_zone: Optional[str] = None  # e.g. "Premium OB on H1"
    entry_price: float
    exit_price: Optional[float] = None
    stop_loss: float
    take_profit: float
    position_size: float = 1.0
    risk_amount: Optional[float] = None
    profit_loss: Optional[float] = None
    status: str = "OPEN"  # OPEN, CLOSED
    notes: Optional[str] = None
    ai_coach: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None


class TradeJournalCreate(BaseModel):
    instrument: str
    direction: str
    session: str
    entry_model: str
    entry_zone: Optional[str] = None
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float = 1.0
    risk_amount: Optional[float] = None
    notes: Optional[str] = None


class TradeJournalUpdate(BaseModel):
    exit_price: Optional[float] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class RiskCalculation(BaseModel):
    account_balance: float
    risk_percentage: float
    entry_price: float
    stop_loss: float
    symbol: str


class AIVerdictRequest(BaseModel):
    symbol: str


# ============== MARKET DATA (Yahoo Finance for all 8 instruments) ==============

market_cache: Dict[str, Any] = {"data": [], "last_update": None}

async def fetch_yahoo_quote(client_session: httpx.AsyncClient, instrument: Dict) -> Optional[Dict]:
    """Fetch a single instrument quote from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{instrument['yahoo']}"
        resp = await client_session.get(
            url,
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        meta = result[0].get("meta", {})
        price = meta.get("regularMarketPrice")
        if price is None:
            return None
        prev = meta.get("chartPreviousClose") or meta.get("previousClose") or price
        change = price - prev
        change_pct = (change / prev * 100) if prev else 0.0
        return {
            "symbol": instrument["symbol"],
            "name": instrument["name"],
            "asset_class": instrument["asset_class"],
            "price": round(price, 5 if instrument["asset_class"] == "forex" else 2),
            "change": round(change, 5 if instrument["asset_class"] == "forex" else 2),
            "change_percent": round(change_pct, 3),
            "high": round(meta.get("regularMarketDayHigh", price), 5 if instrument["asset_class"] == "forex" else 2),
            "low": round(meta.get("regularMarketDayLow", price), 5 if instrument["asset_class"] == "forex" else 2),
            "previous_close": round(prev, 5 if instrument["asset_class"] == "forex" else 2),
            "source": "Yahoo Finance",
        }
    except Exception as e:
        logger.error(f"Yahoo fetch error for {instrument['symbol']}: {e}")
        return None


async def fetch_all_instruments(force: bool = False) -> List[Dict]:
    now = datetime.now(timezone.utc)
    if (not force and market_cache["last_update"] and
            (now - market_cache["last_update"]).total_seconds() < 30 and market_cache["data"]):
        return market_cache["data"]

    async with httpx.AsyncClient() as session:
        tasks = [fetch_yahoo_quote(session, inst) for inst in INSTRUMENTS]
        results = await asyncio.gather(*tasks)

    data = [r for r in results if r]
    if data:
        market_cache["data"] = data
        market_cache["last_update"] = now
    return data if data else market_cache["data"]


async def fetch_yahoo_historical(symbol: str, range_str: str = "1mo", interval: str = "1d") -> List[Dict]:
    """Historical candles. range: 1d, 7d, 1mo. interval: 5m/1h/1d."""
    inst = SYMBOL_LOOKUP.get(symbol)
    if not inst:
        return []
    try:
        async with httpx.AsyncClient(timeout=15.0) as session:
            resp = await session.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{inst['yahoo']}",
                params={"interval": interval, "range": range_str},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return []
            r0 = result[0]
            timestamps = r0.get("timestamp", []) or []
            quote = (r0.get("indicators", {}).get("quote") or [{}])[0]
            closes = quote.get("close", []) or []
            highs = quote.get("high", []) or []
            lows = quote.get("low", []) or []
            opens = quote.get("open", []) or []
            candles = []
            for ts, o, h, lo, c in zip(timestamps, opens, highs, lows, closes):
                if c is None:
                    continue
                candles.append({
                    "timestamp": ts * 1000,
                    "open": round(o or c, 5),
                    "high": round(h or c, 5),
                    "low": round(lo or c, 5),
                    "close": round(c, 5),
                })
            return candles
    except Exception as e:
        logger.error(f"Yahoo historical error for {symbol}: {e}")
        return []


# ============== TECHNICAL INDICATORS ==============

def calc_rsi(prices: List[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return round(100 - (100 / (1 + rs)), 2)


def calc_sma(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


# ============== MARKET ENDPOINTS ==============

@api_router.get("/instruments")
async def get_instruments():
    return {"instruments": INSTRUMENTS, "count": len(INSTRUMENTS)}


@api_router.get("/market/all")
async def get_all_market():
    data = await fetch_all_instruments()
    if not data:
        raise HTTPException(503, "Unable to fetch market data from Yahoo Finance")
    return {
        "instruments": data,
        "source": "Yahoo Finance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(data),
    }


@api_router.get("/market/{symbol:path}/historical")
async def get_historical(symbol: str, range: str = Query(default="1mo")):
    """range: 1d (5m bars), 7d (1h bars), 1mo (1d bars)"""
    symbol = symbol.replace("%2F", "/")
    interval_map = {"1d": "5m", "7d": "1h", "1mo": "1d"}
    interval = interval_map.get(range, "1d")
    yahoo_range = "1d" if range == "1d" else ("5d" if range == "7d" else "1mo")
    candles = await fetch_yahoo_historical(symbol, yahoo_range, interval)
    if not candles:
        raise HTTPException(404, f"No historical data for {symbol}")
    return {"symbol": symbol, "range": range, "candles": candles, "source": "Yahoo Finance"}


# ============== SESSIONS ==============

@api_router.get("/sessions/status")
async def get_sessions_status():
    now = datetime.now(timezone.utc)
    hour = now.hour
    weekday = now.weekday()
    is_weekend = weekday >= 5

    sessions = {
        "sydney":   {"open_utc": 22, "close_utc": 7,  "status": "closed"},
        "tokyo":    {"open_utc": 0,  "close_utc": 9,  "status": "closed"},
        "london":   {"open_utc": 8,  "close_utc": 17, "status": "closed"},
        "new_york": {"open_utc": 13, "close_utc": 22, "status": "closed"},
    }

    if not is_weekend:
        if hour >= 22 or hour < 7:
            sessions["sydney"]["status"] = "open"
        if 0 <= hour < 9:
            sessions["tokyo"]["status"] = "open"
        if 8 <= hour < 17:
            sessions["london"]["status"] = "open"
        if 13 <= hour < 22:
            sessions["new_york"]["status"] = "open"

    overlaps = []
    if sessions["london"]["status"] == "open" and sessions["new_york"]["status"] == "open":
        overlaps.append("London / New York")
    if sessions["tokyo"]["status"] == "open" and sessions["sydney"]["status"] == "open":
        overlaps.append("Sydney / Tokyo")

    return {
        "sessions": sessions,
        "active_overlaps": overlaps,
        "is_weekend": is_weekend,
        "current_time_utc": now.isoformat(),
    }


# ============== ECONOMIC CALENDAR (Forex Factory, 2/3-star only) ==============

RELEVANT_CURRENCIES = {"USD", "EUR", "JPY", "AUD", "GBP"}  # tied to the 8 instruments

calendar_cache = {"events": [], "last_update": None}


async def fetch_forex_factory_calendar() -> List[Dict]:
    try:
        async with httpx.AsyncClient(timeout=15.0) as session:
            resp = await session.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                return []
            raw = resp.json()
            impact_map = {"High": 3, "Medium": 2, "Low": 1, "Holiday": 0}
            events = []
            for item in raw:
                stars = impact_map.get(item.get("impact", "Low"), 1)
                if stars < 2:
                    continue
                country = item.get("country", "USD")
                if country not in RELEVANT_CURRENCIES:
                    continue
                events.append({
                    "id": f"ff-{abs(hash(item.get('title', '') + item.get('date', '')))}",
                    "title": item.get("title", "Event"),
                    "date": item.get("date", ""),
                    "stars": stars,
                    "impact": "HIGH" if stars == 3 else "MEDIUM",
                    "currency": country,
                    "forecast": item.get("forecast"),
                    "previous": item.get("previous"),
                    "actual": item.get("actual"),
                    "source": "Forex Factory",
                })
            return events
    except Exception as e:
        logger.error(f"Forex Factory error: {e}")
        return []


@api_router.get("/calendar/events")
async def get_calendar_events():
    now = datetime.now(timezone.utc)
    if (calendar_cache["last_update"] and
            (now - calendar_cache["last_update"]).total_seconds() < 1800 and
            calendar_cache["events"]):
        return {"events": calendar_cache["events"], "source": "Forex Factory (cached)", "count": len(calendar_cache["events"])}

    events = await fetch_forex_factory_calendar()
    # Filter to future + last 2h
    filtered = []
    for e in events:
        try:
            dt = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
            if dt >= now - timedelta(hours=2):
                filtered.append(e)
        except Exception:
            continue
    filtered.sort(key=lambda x: x.get("date", ""))
    calendar_cache["events"] = filtered
    calendar_cache["last_update"] = now
    return {"events": filtered, "source": "Forex Factory (Live)", "count": len(filtered), "high_impact": len([e for e in filtered if e["stars"] == 3])}


# ============== RISK CALCULATOR ==============

@api_router.post("/risk/calculate")
async def calculate_risk(data: RiskCalculation):
    risk_amount = data.account_balance * (data.risk_percentage / 100)
    pip_risk = abs(data.entry_price - data.stop_loss)
    if pip_risk == 0:
        raise HTTPException(400, "Stop loss cannot equal entry price")
    position_size = risk_amount / pip_risk
    return {
        "risk_amount": round(risk_amount, 2),
        "position_size": round(position_size, 4),
        "pip_risk": round(pip_risk, 5),
        "potential_loss": round(risk_amount, 2),
        "potential_profit_2r": round(risk_amount * 2, 2),
        "potential_profit_3r": round(risk_amount * 3, 2),
        "max_position_value": round(position_size * data.entry_price, 2),
    }


# ============== AI ENGINE (Claude Sonnet 4.5) ==============

def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


async def _llm_chat(system_message: str, user_text: str, session_id: str) -> str:
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system_message,
    ).with_model("anthropic", "claude-sonnet-4-5-20250929")
    return await chat.send_message(UserMessage(text=user_text))


async def get_strategy_rules() -> str:
    doc = await db.config.find_one({"key": "strategy_rules"}, {"_id": 0})
    if doc and doc.get("value"):
        return doc["value"]
    return DEFAULT_STRATEGY_RULES


@api_router.get("/ai/rules")
async def get_rules():
    rules = await get_strategy_rules()
    return {"rules": rules}


class RulesUpdate(BaseModel):
    rules: str


@api_router.put("/ai/rules")
async def update_rules(payload: RulesUpdate):
    await db.config.update_one(
        {"key": "strategy_rules"},
        {"$set": {"value": payload.rules, "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )
    return {"message": "Strategy rules updated", "rules": payload.rules}


@api_router.get("/ai/verdict/{symbol:path}")
async def ai_verdict(symbol: str):
    symbol = symbol.replace("%2F", "/")
    inst = SYMBOL_LOOKUP.get(symbol)
    if not inst:
        raise HTTPException(404, f"Unknown instrument: {symbol}")

    # Get live data + historical
    all_data = await fetch_all_instruments()
    quote = next((q for q in all_data if q["symbol"] == symbol), None)
    if not quote:
        raise HTTPException(503, f"Live data unavailable for {symbol}")

    candles = await fetch_yahoo_historical(symbol, "5d", "1h")
    closes = [c["close"] for c in candles[-100:]] if candles else [quote["price"]]
    rsi = calc_rsi(closes) if len(closes) > 14 else 50.0
    sma20 = calc_sma(closes, 20) or quote["price"]
    sma50 = calc_sma(closes, 50) or quote["price"]
    recent_high = max(closes[-50:]) if len(closes) >= 50 else quote["price"]
    recent_low = min(closes[-50:]) if len(closes) >= 50 else quote["price"]

    rules = await get_strategy_rules()

    user_prompt = f"""Analyze {symbol} ({inst['name']}) and provide a trading verdict.

LIVE MARKET DATA:
- Current Price: {quote['price']}
- 24h Change: {quote['change_percent']:+.2f}%
- Day High: {quote['high']}
- Day Low: {quote['low']}
- Previous Close: {quote['previous_close']}

TECHNICAL CONTEXT (last 100 1h bars):
- RSI(14): {rsi}
- SMA(20): {round(sma20, 5)}
- SMA(50): {round(sma50, 5)}
- Recent Range High: {round(recent_high, 5)}
- Recent Range Low: {round(recent_low, 5)}
- Trend: {"Bullish (SMA20>SMA50)" if sma20 > sma50 else "Bearish (SMA20<SMA50)"}

Return a JSON object with this exact schema (no markdown fences):
{{
  "action": "BUY" | "SELL" | "HOLD" | "WATCH",
  "confidence": <integer 0-100>,
  "reasoning": "<2-3 sentence professional explanation referencing the strategy rules and current conditions>",
  "key_levels": {{"entry": <number>, "stop_loss": <number>, "take_profit": <number>}},
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL"
}}"""

    system = f"You are EliAI's trading engine. Apply this strategy framework strictly:\n\n{rules}\n\nRespond ONLY with valid JSON matching the requested schema."

    try:
        raw = await _llm_chat(system, user_prompt, f"verdict-{symbol}-{int(datetime.now().timestamp())}")
        parsed = json.loads(_strip_json_fences(raw))
    except Exception as e:
        logger.error(f"AI verdict error for {symbol}: {e}")
        # Deterministic fallback so the UI never breaks
        bias = "BULLISH" if sma20 > sma50 else "BEARISH"
        action = "WATCH"
        if rsi < 30 and bias == "BULLISH":
            action = "BUY"
        elif rsi > 70 and bias == "BEARISH":
            action = "SELL"
        parsed = {
            "action": action,
            "confidence": 55,
            "reasoning": f"Fallback: SMA20 {'>' if sma20 > sma50 else '<'} SMA50 with RSI at {rsi}. AI engine temporarily unavailable.",
            "key_levels": {"entry": quote["price"], "stop_loss": round(recent_low, 5), "take_profit": round(recent_high, 5)},
            "bias": bias,
        }

    verdict_doc = {
        "symbol": symbol,
        "asset_class": inst["asset_class"],
        "price_at_verdict": quote["price"],
        "verdict": parsed,
        "rsi": rsi,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.verdicts.insert_one(verdict_doc.copy())
    verdict_doc.pop("_id", None)
    return verdict_doc


@api_router.get("/ai/verdicts")
async def all_verdicts():
    """Return latest cached verdict for each instrument (does NOT regenerate)."""
    results = []
    for inst in INSTRUMENTS:
        doc = await db.verdicts.find_one(
            {"symbol": inst["symbol"]},
            {"_id": 0},
            sort=[("created_at", -1)],
        )
        if doc:
            results.append(doc)
    return {"verdicts": results, "count": len(results)}


@api_router.post("/ai/verdicts/refresh")
async def refresh_all_verdicts():
    """Regenerate verdicts for all 8 instruments."""
    results = []
    for inst in INSTRUMENTS:
        try:
            v = await ai_verdict(inst["symbol"])
            results.append(v)
        except Exception as e:
            logger.error(f"verdict refresh failed for {inst['symbol']}: {e}")
    return {"verdicts": results, "count": len(results), "refreshed_at": datetime.now(timezone.utc).isoformat()}


# ============== TRADE JOURNAL ==============

def _parse_dt(v):
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return v
    return v


@api_router.post("/trades", response_model=TradeJournal)
async def create_trade(trade: TradeJournalCreate):
    obj = TradeJournal(**trade.model_dump())
    doc = obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    if doc.get("closed_at"):
        doc["closed_at"] = doc["closed_at"].isoformat()
    await db.trades.insert_one(doc.copy())
    return obj


@api_router.get("/trades", response_model=List[TradeJournal])
async def list_trades(status: Optional[str] = None, limit: int = 200):
    query = {"status": status} if status else {}
    trades = await db.trades.find(query, {"_id": 0}).sort("created_at", -1).to_list(limit)
    for t in trades:
        t["created_at"] = _parse_dt(t.get("created_at"))
        if t.get("closed_at"):
            t["closed_at"] = _parse_dt(t["closed_at"])
    return trades


@api_router.get("/trades/stats")
async def get_trade_stats():
    pipeline = [
        {"$addFields": {"_pl": {"$ifNull": ["$profit_loss", 0]}}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "open": {"$sum": {"$cond": [{"$eq": ["$status", "OPEN"]}, 1, 0]}},
            "closed": {"$sum": {"$cond": [{"$eq": ["$status", "CLOSED"]}, 1, 0]}},
            "pnl_total": {"$sum": "$_pl"},
            "wins": {"$sum": {"$cond": [{"$gt": ["$_pl", 0]}, 1, 0]}},
            "profits": {"$push": {"$cond": [{"$gt": ["$_pl", 0]}, "$_pl", "$$REMOVE"]}},
            "losses": {"$push": {"$cond": [{"$lt": ["$_pl", 0]}, "$_pl", "$$REMOVE"]}},
        }}
    ]
    result = await db.trades.aggregate(pipeline).to_list(1)
    if not result:
        return {"total_trades": 0, "open_trades": 0, "closed_trades": 0, "win_rate": 0,
                "total_profit_loss": 0, "average_profit": 0, "average_loss": 0, "best_trade": 0, "worst_trade": 0}
    s = result[0]
    profits = s.get("profits", [])
    losses = s.get("losses", [])
    closed = s.get("closed", 0)
    return {
        "total_trades": s.get("total", 0),
        "open_trades": s.get("open", 0),
        "closed_trades": closed,
        "win_rate": round(s.get("wins", 0) / closed * 100, 1) if closed else 0,
        "total_profit_loss": round(s.get("pnl_total", 0), 2),
        "average_profit": round(sum(profits) / len(profits), 2) if profits else 0,
        "average_loss": round(sum(losses) / len(losses), 2) if losses else 0,
        "best_trade": round(max(profits), 2) if profits else 0,
        "worst_trade": round(min(losses), 2) if losses else 0,
    }


@api_router.put("/trades/{trade_id}")
async def update_trade(trade_id: str, update: TradeJournalUpdate):
    update_data = {k: v for k, v in update.model_dump().items() if v is not None}

    if update_data.get("status") == "CLOSED" and update_data.get("exit_price") is not None:
        trade = await db.trades.find_one({"id": trade_id}, {"_id": 0})
        if not trade:
            raise HTTPException(404, "Trade not found")
        entry = trade.get("entry_price", 0)
        exit_price = update_data["exit_price"]
        position = trade.get("position_size", 1.0) or 1.0
        direction = trade.get("direction", "LONG")
        pnl = (exit_price - entry) * position if direction == "LONG" else (entry - exit_price) * position
        update_data["profit_loss"] = round(pnl, 2)
        update_data["closed_at"] = datetime.now(timezone.utc).isoformat()

    result = await db.trades.update_one({"id": trade_id}, {"$set": update_data})
    if result.matched_count == 0:
        raise HTTPException(404, "Trade not found")
    return {"message": "Trade updated", "trade_id": trade_id}


@api_router.delete("/trades/{trade_id}")
async def delete_trade(trade_id: str):
    res = await db.trades.delete_one({"id": trade_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Trade not found")
    return {"message": "Trade deleted"}


# ============== AI COACH ==============

@api_router.post("/ai/coach/score/{trade_id}")
async def coach_score(trade_id: str):
    trade = await db.trades.find_one({"id": trade_id}, {"_id": 0})
    if not trade:
        raise HTTPException(404, "Trade not found")

    rules = await get_strategy_rules()

    user_prompt = f"""Score this CLOSED trade against the strategy rules and provide structured coaching feedback.

TRADE DETAILS:
- Instrument: {trade.get('instrument')}
- Direction: {trade.get('direction')}
- Session: {trade.get('session')}
- Entry Model: {trade.get('entry_model')}
- Entry Zone: {trade.get('entry_zone') or 'N/A'}
- Entry: {trade.get('entry_price')}
- Stop Loss: {trade.get('stop_loss')}
- Take Profit: {trade.get('take_profit')}
- Exit: {trade.get('exit_price')}
- P/L: {trade.get('profit_loss')}
- Trader Notes: {trade.get('notes') or 'None'}

Return JSON ONLY (no markdown) matching:
{{
  "scores": {{
    "setup_quality": <0-10>,
    "entry_execution": <0-10>,
    "risk_management": <0-10>,
    "patience_discipline": <0-10>
  }},
  "overall_score": <0-10>,
  "what_you_did_well": "<one concise sentence>",
  "areas_to_improve": "<one concise sentence>",
  "key_lesson": "<one concise sentence>"
}}"""

    system = f"You are EliAI's Trade Coach. Use this framework to score and coach:\n\n{rules}\n\nBe direct, honest, and constructive. Respond with valid JSON only."

    try:
        raw = await _llm_chat(system, user_prompt, f"coach-{trade_id}")
        parsed = json.loads(_strip_json_fences(raw))
    except Exception as e:
        logger.error(f"AI coach error for {trade_id}: {e}")
        # Fallback heuristic
        pnl = trade.get("profit_loss") or 0
        risk = abs((trade.get("entry_price") or 0) - (trade.get("stop_loss") or 0))
        won = pnl > 0
        parsed = {
            "scores": {
                "setup_quality": 6 if won else 4,
                "entry_execution": 6 if won else 4,
                "risk_management": 7 if risk > 0 else 3,
                "patience_discipline": 6,
            },
            "overall_score": 6 if won else 4,
            "what_you_did_well": "Trade was logged with a defined stop and target." if risk > 0 else "Trade was executed and logged.",
            "areas_to_improve": "AI coach unavailable — record session context and entry trigger for deeper review.",
            "key_lesson": "Detailed notes unlock better feedback. Keep logging.",
        }

    await db.trades.update_one({"id": trade_id}, {"$set": {"ai_coach": parsed}})
    return {"trade_id": trade_id, "ai_coach": parsed}


@api_router.get("/ai/coach/patterns")
async def coach_patterns(limit: int = 10):
    """AI summary of patterns across the last N closed trades."""
    trades = await db.trades.find(
        {"status": "CLOSED"}, {"_id": 0}
    ).sort("closed_at", -1).to_list(limit)

    if len(trades) < 2:
        return {"trades_analyzed": len(trades), "patterns": "Log at least 2 closed trades to see pattern insight.", "habits": []}

    summary = [{
        "instrument": t.get("instrument"),
        "direction": t.get("direction"),
        "session": t.get("session"),
        "entry_model": t.get("entry_model"),
        "pnl": t.get("profit_loss"),
        "coach_score": (t.get("ai_coach") or {}).get("overall_score"),
    } for t in trades]

    user_prompt = f"""Analyze these {len(summary)} recent closed trades and identify the dominant behavioural patterns.

TRADES (most recent first):
{json.dumps(summary, indent=2)}

Return JSON (no markdown):
{{
  "patterns": "<2-3 sentence summary of dominant patterns>",
  "habits": ["<habit 1>", "<habit 2>", "<habit 3>"],
  "top_strength": "<one sentence>",
  "top_weakness": "<one sentence>"
}}"""

    system = "You are EliAI's behaviour analyst. Identify behavioural patterns across a trader's history. Respond with JSON only."

    try:
        raw = await _llm_chat(system, user_prompt, f"patterns-{int(datetime.now().timestamp())}")
        parsed = json.loads(_strip_json_fences(raw))
    except Exception as e:
        logger.error(f"Coach patterns error: {e}")
        parsed = {
            "patterns": f"Reviewed {len(trades)} trades. AI pattern analysis temporarily unavailable.",
            "habits": [],
            "top_strength": "",
            "top_weakness": "",
        }

    return {"trades_analyzed": len(trades), **parsed}


# ============== EQUITY CURVE ==============

@api_router.get("/reports/equity-curve")
async def equity_curve():
    trades = await db.trades.find(
        {"status": "CLOSED"}, {"_id": 0, "id": 1, "closed_at": 1, "profit_loss": 1, "instrument": 1}
    ).sort("closed_at", 1).to_list(500)

    cumulative = 0.0
    curve = []
    for t in trades:
        pnl = t.get("profit_loss") or 0
        cumulative += pnl
        curve.append({
            "closed_at": t.get("closed_at"),
            "instrument": t.get("instrument"),
            "pnl": pnl,
            "cumulative": round(cumulative, 2),
        })
    return {"curve": curve, "total_pnl": round(cumulative, 2), "trades": len(curve)}


# ============== EMAIL ALERTS (open-source SMTP via stdlib) ==============
#
# SMTP is configured purely via env vars — no SDK / no third-party lock-in.
# Set these in /app/backend/.env to enable real sending:
#   SMTP_HOST=smtp.gmail.com
#   SMTP_PORT=587
#   SMTP_USER=alerts@eliai.trade
#   SMTP_PASS=<16-char-app-password>
#   SMTP_FROM=alerts@eliai.trade
#   SMTP_TLS=1                       # 1 = STARTTLS (default), 0 = plain, ssl = SMTP_SSL
#   ALERT_LEAD_MINUTES=30            # how many minutes before event to fire (default 30)
#
# Until SMTP_HOST is set the system stores recipients, renders previews, and
# logs "would-send" messages — letting you swap in real credentials later
# with zero code changes.

ALERT_LEAD_MINUTES = int(os.environ.get('ALERT_LEAD_MINUTES', '30'))


class AlertRecipient(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RecipientCreate(BaseModel):
    email: str
    enabled: bool = True


def smtp_configured() -> bool:
    return bool(os.environ.get('SMTP_HOST') and os.environ.get('SMTP_FROM'))


def render_event_email(event: Dict) -> Dict[str, str]:
    """Build subject + HTML body for a calendar event alert."""
    stars = event.get("stars", 2)
    star_str = "★" * stars
    ccy = event.get("currency", "")
    title = event.get("title", "Event")
    try:
        dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        when = dt.strftime("%a %d %b · %H:%M UTC")
    except Exception:
        when = event.get("date", "")
    subject = f"[EliAI] {star_str} {ccy} · {title} in {ALERT_LEAD_MINUTES} min"
    html = f"""<!doctype html>
<html><body style="font-family: -apple-system, sans-serif; background:#0A1628; color:#F8F9FA; margin:0; padding:24px;">
  <div style="max-width:540px; margin:auto; background:#0E1F36; border:1px solid #1E3A5F; border-radius:6px; padding:24px;">
    <div style="border-bottom:1px solid #1E3A5F; padding-bottom:12px; margin-bottom:16px;">
      <div style="color:#D4AF37; font-size:11px; letter-spacing:2px; font-weight:700;">ELIAI · ECONOMIC ALERT</div>
      <div style="color:#fff; font-size:22px; font-weight:700; margin-top:4px;">{title}</div>
    </div>
    <table style="width:100%; font-size:14px; color:#CBD5E1;">
      <tr><td style="padding:6px 0; color:#94A3B8;">Currency</td><td style="color:#D4AF37; font-weight:600;">{ccy}</td></tr>
      <tr><td style="padding:6px 0; color:#94A3B8;">Impact</td><td style="color:#fff; font-weight:600;">{star_str} ({stars}-star)</td></tr>
      <tr><td style="padding:6px 0; color:#94A3B8;">Release</td><td style="color:#fff; font-weight:600;">{when}</td></tr>
      <tr><td style="padding:6px 0; color:#94A3B8;">Forecast</td><td style="color:#fff;">{event.get("forecast") or "—"}</td></tr>
      <tr><td style="padding:6px 0; color:#94A3B8;">Previous</td><td style="color:#fff;">{event.get("previous") or "—"}</td></tr>
    </table>
    <p style="color:#94A3B8; font-size:12px; margin-top:20px;">
      Fires in <strong style="color:#D4AF37;">{ALERT_LEAD_MINUTES} minutes</strong>.
      Per the EliAI framework, avoid new entries 15 minutes before / after this release for the affected currency and Gold.
    </p>
  </div>
  <div style="text-align:center; color:#475569; font-size:10px; margin-top:16px;">EliAI Trade · AI Trading Intelligence</div>
</body></html>"""
    return {"subject": subject, "html": html}


def send_email_sync(to_list: List[str], subject: str, html: str) -> Dict[str, Any]:
    """Sync SMTP send via stdlib. Returns status dict; never raises."""
    if not smtp_configured():
        logger.info(f"[mailer] SMTP not configured — would have sent '{subject}' to {to_list}")
        return {"sent": False, "reason": "SMTP not configured", "recipients": to_list}
    host = os.environ['SMTP_HOST']
    port = int(os.environ.get('SMTP_PORT', '587'))
    user = os.environ.get('SMTP_USER')
    password = os.environ.get('SMTP_PASS')
    sender = os.environ['SMTP_FROM']
    tls_mode = os.environ.get('SMTP_TLS', '1').lower()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(to_list)
    msg.set_content("Your client does not support HTML. Open the EliAI dashboard for the full alert.")
    msg.add_alternative(html, subtype="html")

    try:
        if tls_mode == "ssl":
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                if user:
                    s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as s:
                if tls_mode == "1":
                    s.starttls()
                if user:
                    s.login(user, password)
                s.send_message(msg)
        return {"sent": True, "recipients": to_list}
    except Exception as e:
        logger.error(f"[mailer] SMTP send failed: {e}")
        return {"sent": False, "reason": str(e), "recipients": to_list}


async def send_email(to_list: List[str], subject: str, html: str) -> Dict[str, Any]:
    """Async wrapper — runs smtplib in a thread to keep the event loop free."""
    return await asyncio.to_thread(send_email_sync, to_list, subject, html)


async def get_active_recipients() -> List[str]:
    docs = await db.alert_recipients.find({"enabled": True}, {"_id": 0, "email": 1}).to_list(200)
    return [d["email"] for d in docs]


async def dispatch_due_alerts() -> Dict[str, Any]:
    """Find 3-star calendar events firing in the next ALERT_LEAD_MINUTES and email them."""
    events_resp = await get_calendar_events()
    events = events_resp.get("events", [])
    now = datetime.now(timezone.utc)
    recipients = await get_active_recipients()
    dispatched = []
    skipped = 0

    for ev in events:
        if ev.get("stars") != 3:
            continue
        try:
            ev_dt = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        except Exception:
            continue
        delta = (ev_dt - now).total_seconds() / 60.0
        if not (ALERT_LEAD_MINUTES - 5 <= delta <= ALERT_LEAD_MINUTES + 5):
            continue
        if not recipients:
            skipped += 1
            continue
        # Idempotency: unique index on event_id prevents duplicate sends
        try:
            await db.alerts_sent.insert_one({
                "event_id": ev["id"], "title": ev["title"], "currency": ev["currency"],
                "stars": ev["stars"], "sent_at": now.isoformat(), "recipients": recipients,
            })
        except Exception:
            skipped += 1
            continue
        rendered = render_event_email(ev)
        result = await send_email(recipients, rendered["subject"], rendered["html"])
        dispatched.append({"event_id": ev["id"], "title": ev["title"], "result": result})

    return {"dispatched": dispatched, "skipped": skipped, "checked_events": len(events),
            "recipients": len(recipients), "smtp_configured": smtp_configured()}


async def alert_scheduler_loop():
    """Background task — checks every 5 minutes."""
    await asyncio.sleep(20)  # warm-up
    while True:
        try:
            await dispatch_due_alerts()
        except Exception as e:
            logger.error(f"[scheduler] dispatch error: {e}")
        await asyncio.sleep(300)  # 5 min


# ---------- Alert endpoints ----------

@api_router.get("/alerts/config")
async def alerts_config():
    return {
        "smtp_configured": smtp_configured(),
        "smtp_host": os.environ.get('SMTP_HOST') or None,
        "smtp_from": os.environ.get('SMTP_FROM') or None,
        "lead_minutes": ALERT_LEAD_MINUTES,
        "scheduler_running": True,
        "setup_hint": "Set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM in /app/backend/.env",
    }


@api_router.get("/alerts/recipients")
async def list_recipients():
    docs = await db.alert_recipients.find({}, {"_id": 0}).sort("created_at", 1).to_list(200)
    for d in docs:
        if isinstance(d.get("created_at"), str):
            d["created_at"] = _parse_dt(d["created_at"])
    return {"recipients": docs, "count": len(docs)}


@api_router.post("/alerts/recipients", response_model=AlertRecipient)
async def add_recipient(payload: RecipientCreate):
    email = payload.email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(400, "Invalid email")
    existing = await db.alert_recipients.find_one({"email": email}, {"_id": 0})
    if existing:
        raise HTTPException(409, "Recipient already exists")
    obj = AlertRecipient(email=email, enabled=payload.enabled)
    doc = obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.alert_recipients.insert_one(doc.copy())
    return obj


@api_router.put("/alerts/recipients/{recipient_id}")
async def toggle_recipient(recipient_id: str, payload: RecipientCreate):
    res = await db.alert_recipients.update_one(
        {"id": recipient_id},
        {"$set": {"enabled": payload.enabled}},
    )
    if res.matched_count == 0:
        raise HTTPException(404, "Recipient not found")
    return {"id": recipient_id, "enabled": payload.enabled}


@api_router.delete("/alerts/recipients/{recipient_id}")
async def delete_recipient(recipient_id: str):
    res = await db.alert_recipients.delete_one({"id": recipient_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "Recipient not found")
    return {"message": "Recipient removed"}


@api_router.post("/alerts/preview")
async def preview_alert():
    """Render a sample 3-star alert email body for the next eligible event."""
    events_resp = await get_calendar_events()
    events = events_resp.get("events", [])
    sample = next((e for e in events if e.get("stars") == 3), None)
    if not sample:
        sample = {
            "id": "sample", "title": "Non-Farm Payrolls",
            "currency": "USD", "stars": 3,
            "date": (datetime.now(timezone.utc) + timedelta(minutes=ALERT_LEAD_MINUTES)).isoformat(),
            "forecast": "180K", "previous": "256K",
        }
    rendered = render_event_email(sample)
    return {"event": sample, "subject": rendered["subject"], "html": rendered["html"]}


@api_router.post("/alerts/test-send")
async def test_send():
    """Sends a test email to all active recipients (or simulates if SMTP unset)."""
    recipients = await get_active_recipients()
    if not recipients:
        raise HTTPException(400, "No active recipients configured")
    rendered = render_event_email({
        "id": "test", "title": "EliAI Alert Test",
        "currency": "USD", "stars": 3,
        "date": datetime.now(timezone.utc).isoformat(),
        "forecast": "N/A", "previous": "N/A",
    })
    result = await send_email(recipients, "[EliAI] Test alert", rendered["html"])
    return result


@api_router.post("/alerts/dispatch-now")
async def dispatch_now():
    """Manually trigger the alert dispatcher (admin/testing)."""
    return await dispatch_due_alerts()


@api_router.get("/alerts/history")
async def alerts_history(limit: int = 50):
    docs = await db.alerts_sent.find({}, {"_id": 0}).sort("sent_at", -1).to_list(limit)
    return {"history": docs, "count": len(docs)}


# ============== HEALTH ==============

@api_router.get("/")
async def root():
    return {"name": "EliAI Trade API", "version": "2.0.0", "instruments": len(INSTRUMENTS)}


# ============== APP WIRING ==============

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try:
        await db.trades.create_index("created_at")
        await db.trades.create_index([("status", 1), ("created_at", -1)])
        await db.verdicts.create_index([("symbol", 1), ("created_at", -1)])
        await db.alert_recipients.create_index("email", unique=True)
        await db.alerts_sent.create_index("event_id", unique=True)
        # Drop legacy collections from the crypto era (idempotent)
        await db.trade_signals.drop()
        await db.status_checks.drop()
        # Seed default alert recipients if empty
        if await db.alert_recipients.count_documents({}) == 0:
            for email in ["brodie@eliai.trade", "mark@eliai.trade", "brooklyn@eliai.trade"]:
                await db.alert_recipients.insert_one({
                    "id": str(uuid.uuid4()), "email": email, "enabled": True,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
        # Spawn the alert scheduler (every 5 min)
        asyncio.create_task(alert_scheduler_loop(), name="alert-scheduler")
        logger.info("EliAI Trade API ready — indexes created, alert scheduler started")
    except Exception as e:
        logger.warning(f"Startup warning: {e}")


@app.on_event("shutdown")
async def shutdown():
    # Cancel the alert scheduler task gracefully
    for task in asyncio.all_tasks():
        if task.get_name() == "alert-scheduler":
            task.cancel()
    client.close()
