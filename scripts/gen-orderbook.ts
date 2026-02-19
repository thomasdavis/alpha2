/**
 * Generate a simulated financial orderbook dataset.
 * 50,000 lines of realistic-looking order flow.
 *
 * Usage: npx tsx scripts/gen-orderbook.ts
 */

// ── Seeded RNG (deterministic output) ─────────────────────────────────────

let seed = 1337;
function rand(): number {
  seed = (seed * 1664525 + 1013904223) & 0x7fffffff;
  return seed / 0x7fffffff;
}
function randInt(min: number, max: number): number {
  return Math.floor(rand() * (max - min + 1)) + min;
}
function pick<T>(arr: T[]): T {
  return arr[randInt(0, arr.length - 1)];
}
function weightedPick<T>(items: T[], weights: number[]): T {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rand() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}
function gaussian(): number {
  let u = 0, v = 0;
  while (u === 0) u = rand();
  while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── Market config ─────────────────────────────────────────────────────────

const TICKERS = [
  { sym: "AAPL", base: 187.50, vol: 0.015, avgSize: 200, sector: "tech" },
  { sym: "MSFT", base: 415.20, vol: 0.012, avgSize: 150, sector: "tech" },
  { sym: "GOOGL", base: 141.80, vol: 0.014, avgSize: 180, sector: "tech" },
  { sym: "AMZN", base: 178.90, vol: 0.016, avgSize: 250, sector: "tech" },
  { sym: "NVDA", base: 721.30, vol: 0.022, avgSize: 300, sector: "tech" },
  { sym: "META", base: 484.10, vol: 0.018, avgSize: 170, sector: "tech" },
  { sym: "TSLA", base: 193.60, vol: 0.028, avgSize: 400, sector: "auto" },
  { sym: "JPM", base: 183.40, vol: 0.010, avgSize: 120, sector: "finance" },
  { sym: "BAC", base: 33.80, vol: 0.011, avgSize: 500, sector: "finance" },
  { sym: "GS", base: 387.20, vol: 0.013, avgSize: 80, sector: "finance" },
  { sym: "WMT", base: 168.50, vol: 0.008, avgSize: 130, sector: "retail" },
  { sym: "XOM", base: 104.70, vol: 0.012, avgSize: 200, sector: "energy" },
  { sym: "CVX", base: 152.30, vol: 0.011, avgSize: 160, sector: "energy" },
  { sym: "PFE", base: 27.40, vol: 0.013, avgSize: 600, sector: "pharma" },
  { sym: "UNH", base: 527.80, vol: 0.009, avgSize: 60, sector: "health" },
  { sym: "SPY", base: 497.20, vol: 0.007, avgSize: 500, sector: "etf" },
  { sym: "QQQ", base: 421.60, vol: 0.009, avgSize: 350, sector: "etf" },
  { sym: "IWM", base: 198.30, vol: 0.011, avgSize: 280, sector: "etf" },
  { sym: "BTC", base: 51240.00, vol: 0.025, avgSize: 2, sector: "crypto" },
  { sym: "ETH", base: 2890.00, vol: 0.030, avgSize: 10, sector: "crypto" },
];

const VENUES = ["NYSE", "NASDAQ", "ARCA", "BATS", "IEX", "EDGX", "MEMX", "LTSE"];
const CRYPTO_VENUES = ["COINBASE", "BINANCE", "KRAKEN", "GEMINI"];
const ORDER_TYPES = ["LIMIT", "MARKET", "STOP", "STOP_LIMIT", "IOC", "FOK", "MOC"];
const ORDER_TYPE_WEIGHTS = [45, 25, 8, 5, 10, 4, 3];
const TIF = ["DAY", "GTC", "IOC", "FOK", "GTD", "OPG", "CLO"];
const TIF_WEIGHTS = [40, 25, 15, 5, 8, 4, 3];
const ACTIONS = ["NEW", "NEW", "NEW", "NEW", "CANCEL", "CANCEL", "MODIFY", "FILL", "FILL", "FILL", "PARTIAL_FILL", "PARTIAL_FILL", "REJECT", "EXPIRED"];
const REJECT_REASONS = [
  "INSUFFICIENT_FUNDS", "PRICE_OUT_OF_RANGE", "SIZE_EXCEEDS_LIMIT",
  "MARKET_CLOSED", "SYMBOL_HALTED", "DUPLICATE_ORDER_ID",
  "INVALID_PRICE_INCREMENT", "SELF_TRADE_PREVENTION",
];
const COND_CODES = ["", "", "", "", "@", "F", "I", "W", "T", "U", "Z"];
const TRADER_PREFIXES = ["HFT", "MM", "INST", "RET", "ALGO", "ARB", "DARK", "SWEEP"];

// Track prices per ticker for realistic walk
const prices: Record<string, number> = {};
const bids: Record<string, number> = {};
const asks: Record<string, number> = {};
for (const t of TICKERS) {
  prices[t.sym] = t.base;
  const spread = t.base * 0.0005;
  bids[t.sym] = t.base - spread / 2;
  asks[t.sym] = t.base + spread / 2;
}

// ── Time simulation ───────────────────────────────────────────────────────

// Start at 2024-02-19 09:30:00 ET (market open)
let ts = new Date("2024-02-19T09:30:00.000-05:00").getTime();
const MARKET_OPEN_MS = ts;
const MARKET_CLOSE_MS = new Date("2024-02-19T16:00:00.000-05:00").getTime();
const TRADING_DAY_MS = MARKET_CLOSE_MS - MARKET_OPEN_MS;

function advanceTime(): void {
  // Variable interval: faster at open/close, slower midday
  const elapsed = ts - MARKET_OPEN_MS;
  const pctDay = elapsed / TRADING_DAY_MS;

  let baseIntervalMs: number;
  if (pctDay < 0.05) baseIntervalMs = 50;       // First 20 min: very fast
  else if (pctDay < 0.15) baseIntervalMs = 150;  // Morning rush
  else if (pctDay > 0.95) baseIntervalMs = 60;   // Close auction
  else if (pctDay > 0.85) baseIntervalMs = 120;  // Afternoon pickup
  else baseIntervalMs = 300;                      // Midday lull

  const jitter = randInt(1, Math.floor(baseIntervalMs * 1.5));
  ts += jitter;

  // Wrap to next day if past close
  if (ts > MARKET_CLOSE_MS) {
    ts = MARKET_OPEN_MS + 86400000 + randInt(0, 60000);
  }
}

function fmtTs(): string {
  const d = new Date(ts);
  return d.toISOString().replace("T", " ").replace("Z", "");
}

function fmtPrice(p: number, isCrypto: boolean): string {
  return isCrypto ? p.toFixed(2) : p.toFixed(2);
}

// ── Order ID tracking ─────────────────────────────────────────────────────

let orderSeq = 100000;
const activeOrders: Map<string, { sym: string; side: string; price: number; qty: number; remaining: number; traderId: string }> = new Map();

function nextOrderId(): string {
  orderSeq++;
  const hex = orderSeq.toString(16).toUpperCase();
  return `ORD-${hex}`;
}

// ── Generate one line ─────────────────────────────────────────────────────

function generateLine(): string {
  const ticker = pick(TICKERS);
  const sym = ticker.sym;
  const isCrypto = ticker.sector === "crypto";
  const venue = isCrypto ? pick(CRYPTO_VENUES) : pick(VENUES);

  // Random walk the price
  const drift = gaussian() * ticker.vol * ticker.base * 0.001;
  prices[sym] += drift;
  if (prices[sym] < ticker.base * 0.9) prices[sym] = ticker.base * 0.9;
  if (prices[sym] > ticker.base * 1.1) prices[sym] = ticker.base * 1.1;

  const mid = prices[sym];
  const spreadBps = isCrypto ? randInt(3, 15) : randInt(1, 8);
  const spread = mid * spreadBps * 0.0001;
  bids[sym] = mid - spread / 2;
  asks[sym] = mid + spread / 2;

  const action = pick(ACTIONS);
  const side = rand() < 0.5 ? "BUY" : "SELL";
  const traderId = `${pick(TRADER_PREFIXES)}-${randInt(1000, 9999)}`;

  // Size: log-normal-ish distribution
  let qty = Math.round(ticker.avgSize * Math.exp(gaussian() * 0.6));
  if (qty < 1) qty = 1;
  if (!isCrypto) qty = Math.round(qty / 10) * 10 || 10; // Round to lots of 10

  let price: number;
  const orderType = weightedPick(ORDER_TYPES, ORDER_TYPE_WEIGHTS);

  if (orderType === "MARKET" || orderType === "MOC") {
    price = side === "BUY" ? asks[sym] : bids[sym];
  } else {
    // Limit orders: some aggressive, some passive, some deep
    const aggressiveness = gaussian() * 0.3;
    if (side === "BUY") {
      price = bids[sym] * (1 + aggressiveness * 0.001);
    } else {
      price = asks[sym] * (1 - aggressiveness * 0.001);
    }
  }

  // Price tick rounding
  const tickSize = isCrypto ? 0.01 : (price > 1 ? 0.01 : 0.0001);
  price = Math.round(price / tickSize) * tickSize;
  if (price <= 0) price = tickSize;

  const tif = weightedPick(TIF, TIF_WEIGHTS);
  const condCode = pick(COND_CODES);

  advanceTime();
  const timestamp = fmtTs();

  // Build line based on action type
  let line: string;

  switch (action) {
    case "NEW": {
      const orderId = nextOrderId();
      activeOrders.set(orderId, { sym, side, price, qty, remaining: qty, traderId });

      // Trim active orders to prevent unbounded growth
      if (activeOrders.size > 2000) {
        const keys = Array.from(activeOrders.keys());
        for (let i = 0; i < 500; i++) activeOrders.delete(keys[i]);
      }

      line = `${timestamp} | ${action} | ${orderId} | ${sym} | ${side} | ${orderType} | ${fmtPrice(price, isCrypto)} | ${qty} | ${tif} | ${venue} | ${traderId}`;
      break;
    }
    case "FILL":
    case "PARTIAL_FILL": {
      const fillOrdId = activeOrders.size > 0 ? pick(Array.from(activeOrders.keys())) : nextOrderId();
      const existing = activeOrders.get(fillOrdId);
      const fillSym = existing?.sym ?? sym;
      const fillSide = existing?.side ?? side;
      const fillTrader = existing?.traderId ?? traderId;

      let fillQty: number;
      let fillPrice: number;

      if (existing) {
        if (action === "PARTIAL_FILL") {
          fillQty = Math.max(1, Math.floor(existing.remaining * (0.1 + rand() * 0.5)));
          existing.remaining -= fillQty;
          if (existing.remaining <= 0) activeOrders.delete(fillOrdId);
        } else {
          fillQty = existing.remaining;
          activeOrders.delete(fillOrdId);
        }
        // Fill price near the order price with slight slippage
        const slippage = gaussian() * 0.0002 * existing.price;
        fillPrice = existing.price + (fillSide === "BUY" ? slippage : -slippage);
      } else {
        fillQty = qty;
        fillPrice = price;
      }

      if (!isCrypto) fillQty = Math.round(fillQty / 10) * 10 || 10;
      fillPrice = Math.round(fillPrice / tickSize) * tickSize;
      if (fillPrice <= 0) fillPrice = tickSize;

      const fillValue = fillPrice * fillQty;
      const execId = `EXE-${(orderSeq++).toString(16).toUpperCase()}`;
      const liq = rand() < 0.4 ? "ADD" : "REMOVE";

      line = `${timestamp} | ${action} | ${fillOrdId} | ${fillSym} | ${fillSide} | ${fmtPrice(fillPrice, isCrypto)} | ${fillQty} | ${fillValue.toFixed(2)} | ${execId} | ${venue} | ${liq} | ${condCode} | ${fillTrader}`;
      break;
    }
    case "CANCEL": {
      const cancelId = activeOrders.size > 0 ? pick(Array.from(activeOrders.keys())) : nextOrderId();
      const existing = activeOrders.get(cancelId);
      const cancelSym = existing?.sym ?? sym;
      const cancelSide = existing?.side ?? side;
      const cancelTrader = existing?.traderId ?? traderId;
      const remaining = existing?.remaining ?? qty;
      activeOrders.delete(cancelId);

      const reason = rand() < 0.3 ? "USER_REQUESTED" : rand() < 0.5 ? "IOC_EXPIRED" : "STRATEGY_CANCEL";
      line = `${timestamp} | ${action} | ${cancelId} | ${cancelSym} | ${cancelSide} | ${remaining} | ${reason} | ${venue} | ${cancelTrader}`;
      break;
    }
    case "MODIFY": {
      const modId = activeOrders.size > 0 ? pick(Array.from(activeOrders.keys())) : nextOrderId();
      const existing = activeOrders.get(modId);
      const modSym = existing?.sym ?? sym;
      const modSide = existing?.side ?? side;
      const modTrader = existing?.traderId ?? traderId;

      const oldPrice = existing?.price ?? price;
      const newPrice = Math.round((oldPrice * (1 + gaussian() * 0.001)) / tickSize) * tickSize;
      const oldQty = existing?.remaining ?? qty;
      const newQty = rand() < 0.5 ? oldQty : Math.max(1, oldQty + Math.round(gaussian() * oldQty * 0.3));

      if (existing) {
        existing.price = newPrice;
        existing.remaining = newQty;
        existing.qty = newQty;
      }

      line = `${timestamp} | ${action} | ${modId} | ${modSym} | ${modSide} | ${fmtPrice(oldPrice, isCrypto)}->${fmtPrice(newPrice, isCrypto)} | ${oldQty}->${newQty} | ${venue} | ${modTrader}`;
      break;
    }
    case "REJECT": {
      const rejId = nextOrderId();
      const reason = pick(REJECT_REASONS);
      line = `${timestamp} | ${action} | ${rejId} | ${sym} | ${side} | ${orderType} | ${fmtPrice(price, isCrypto)} | ${qty} | ${reason} | ${venue} | ${traderId}`;
      break;
    }
    case "EXPIRED": {
      const expId = activeOrders.size > 0 ? pick(Array.from(activeOrders.keys())) : nextOrderId();
      const existing = activeOrders.get(expId);
      const expSym = existing?.sym ?? sym;
      const expSide = existing?.side ?? side;
      const expTrader = existing?.traderId ?? traderId;
      const remaining = existing?.remaining ?? qty;
      activeOrders.delete(expId);

      line = `${timestamp} | ${action} | ${expId} | ${expSym} | ${expSide} | ${remaining} remaining | TIF_EXPIRED | ${venue} | ${expTrader}`;
      break;
    }
    default:
      line = `${timestamp} | UNKNOWN | ${sym}`;
  }

  // Occasionally inject market events
  if (rand() < 0.008) {
    const events = [
      `[HALT] ${sym} trading halted - LULD circuit breaker triggered at ${fmtPrice(mid, isCrypto)}`,
      `[RESUME] ${sym} trading resumed after ${randInt(2, 15)} minute halt`,
      `[IMBALANCE] ${sym} ${side} imbalance ${(qty * randInt(5, 50)).toLocaleString()} shares at ${fmtPrice(mid, isCrypto)} | ${venue}`,
      `[AUCTION] ${sym} opening auction ${(qty * randInt(10, 100)).toLocaleString()} shares crossed at ${fmtPrice(mid, isCrypto)}`,
      `[NBBO] ${sym} ${fmtPrice(bids[sym], isCrypto)} x ${fmtPrice(asks[sym], isCrypto)} spread=${spreadBps}bps depth=${randInt(100, 5000)}x${randInt(100, 5000)}`,
      `[SWEEP] ${sym} ISO sweep ${side} ${qty * randInt(3, 10)} across ${randInt(2, 6)} venues | ${traderId}`,
      `[ODDLOT] ${sym} ${side} ${randInt(1, 9)} shares at ${fmtPrice(price, isCrypto)} | ${venue} | ${traderId}`,
      `[BLOCK] ${sym} block trade ${(qty * randInt(20, 200)).toLocaleString()} shares at ${fmtPrice(mid + gaussian() * spread, isCrypto)} reported late`,
    ];
    line += `\n${timestamp} | ${pick(events)}`;
  }

  return line;
}

// ── Main ──────────────────────────────────────────────────────────────────

const LINES = 50_000;
const outPath = "data/dumb_finance.txt";

const fs = await import("node:fs");
const path = await import("node:path");

fs.mkdirSync(path.dirname(outPath), { recursive: true });
const fd = fs.openSync(outPath, "w");

// Header comment
fs.writeSync(fd, "# DUMB_FINANCE Orderbook Dataset\n");
fs.writeSync(fd, "# Simulated financial order flow — 20 tickers, 8 venues, realistic price dynamics\n");
fs.writeSync(fd, "# Format: TIMESTAMP | ACTION | ORDER_ID | SYMBOL | SIDE | ... (varies by action type)\n");
fs.writeSync(fd, "#\n");

let written = 0;
while (written < LINES) {
  const line = generateLine();
  fs.writeSync(fd, line + "\n");
  // Count actual lines (market events inject extra lines)
  written += line.split("\n").length;
}

fs.closeSync(fd);

const stat = fs.statSync(outPath);
console.log(`Generated ${written} lines → ${outPath} (${(stat.size / 1024).toFixed(0)} KB)`);
