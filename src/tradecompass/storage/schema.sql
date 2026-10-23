-- TradeCompass SQLite schema v1（SPEC §18）
-- 版本号记录在 PRAGMA user_version 中，由 storage/db.py 负责迁移。

CREATE TABLE trades (
  id INTEGER PRIMARY KEY,
  dedupe_key TEXT NOT NULL UNIQUE,
  trade_date TEXT NOT NULL,
  trade_time TEXT,
  symbol TEXT NOT NULL,
  name TEXT,
  side TEXT NOT NULL CHECK (side IN ('buy','sell','dividend','bonus_share','tax')),
  quantity INTEGER NOT NULL,
  price_raw REAL,
  amount REAL,
  fee_commission REAL DEFAULT 0,
  fee_stamp REAL DEFAULT 0,
  fee_transfer REAL DEFAULT 0,
  fee_other REAL DEFAULT 0,
  source TEXT NOT NULL CHECK (source IN ('broker','opening','manual')),
  raw_json TEXT,
  imported_at TEXT NOT NULL
);
CREATE INDEX idx_trades_symbol_date ON trades(symbol, trade_date);

CREATE TABLE import_issues (
  id INTEGER PRIMARY KEY, file TEXT, row_no INTEGER, reason TEXT, raw_json TEXT, created_at TEXT
);

CREATE TABLE trade_notes (
  id INTEGER PRIMARY KEY,
  note_date TEXT NOT NULL, symbol TEXT NOT NULL,
  side TEXT NOT NULL CHECK (side IN ('buy','sell')),
  quantity INTEGER NOT NULL, price_raw REAL NOT NULL,
  tag TEXT NOT NULL, note TEXT,
  trade_ref INTEGER REFERENCES trades(id), matched INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

CREATE TABLE position_meta (
  symbol TEXT PRIMARY KEY,
  category TEXT NOT NULL CHECK (category IN ('base','swing','legacy')),
  legacy_class TEXT CHECK (legacy_class IN ('keep','watch','clean','fragment')),
  legacy_d0 TEXT, legacy_close_d0 REAL, legacy_atr_d0 REAL,
  strategy_id TEXT, entry_date TEXT,
  initial_stop_raw REAL, current_stop_raw REAL, highest_close_raw REAL,
  updated_at TEXT NOT NULL
);

CREATE TABLE signals (
  id INTEGER PRIMARY KEY,
  as_of TEXT NOT NULL, symbol TEXT NOT NULL,
  strategy_id TEXT NOT NULL, params_hash TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('entry','exit')),
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('emitted','blocked','dropped')),
  blocked_by TEXT
);

CREATE TABLE actions (
  id INTEGER PRIMARY KEY,
  as_of TEXT NOT NULL, for_date TEXT NOT NULL,
  symbol TEXT NOT NULL, priority TEXT NOT NULL, kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','done','expired','ignored'))
);

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, symbol TEXT, event TEXT NOT NULL,
  level TEXT NOT NULL, message TEXT NOT NULL, dedupe_key TEXT NOT NULL UNIQUE, pushed INTEGER NOT NULL
);

CREATE TABLE regimes (
  date TEXT PRIMARY KEY, score INTEGER NOT NULL, level TEXT NOT NULL,
  index_bear INTEGER NOT NULL, components_json TEXT NOT NULL
);

CREATE TABLE backtest_runs (
  id INTEGER PRIMARY KEY, strategy_id TEXT NOT NULL, params_hash TEXT NOT NULL,
  sample TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
  passed INTEGER NOT NULL, failed_items TEXT, metrics_json TEXT NOT NULL,
  report_path TEXT NOT NULL, run_at TEXT NOT NULL
);

CREATE TABLE config_versions (
  id INTEGER PRIMARY KEY, file TEXT NOT NULL, content_hash TEXT NOT NULL,
  content TEXT NOT NULL, submitted_at TEXT NOT NULL, effective_from TEXT NOT NULL
);

CREATE TABLE events (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, symbol TEXT, detail_json TEXT
);

CREATE TABLE runs (
  id INTEGER PRIMARY KEY, task TEXT NOT NULL, run_date TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('running','ok','failed')), error TEXT
);
