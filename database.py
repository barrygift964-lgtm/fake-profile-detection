"""SQLite database layer (tables: users, datasets, trained_models, detections)."""
import sqlite3

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    is_active     INTEGER NOT NULL DEFAULT 1,
    failed_logins INTEGER NOT NULL DEFAULT 0,
    date_joined   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_by   INTEGER NOT NULL REFERENCES users(user_id),
    file_name     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    total_records INTEGER NOT NULL,
    genuine_count INTEGER NOT NULL,
    fake_count    INTEGER NOT NULL,
    uploaded_on   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS trained_models (
    model_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL REFERENCES datasets(dataset_id),
    model_path      TEXT NOT NULL,
    n_estimators    INTEGER NOT NULL,
    train_size      INTEGER NOT NULL,
    test_size       INTEGER NOT NULL,
    accuracy        REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score    REAL NOT NULL,
    f1_score        REAL NOT NULL,
    error_rate      REAL NOT NULL,
    tp INTEGER NOT NULL, tn INTEGER NOT NULL, fp INTEGER NOT NULL, fn INTEGER NOT NULL,
    importances     TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1,
    trained_on      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE TABLE IF NOT EXISTS detections (
    detection_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(user_id),
    model_id        INTEGER NOT NULL REFERENCES trained_models(model_id),
    account_name    TEXT,
    statuses_count INTEGER, followers_count INTEGER, friends_count INTEGER,
    favourites_count INTEGER, listed_count INTEGER,
    ff_ratio REAL, url_ratio REAL, average_mentions REAL, average_hashtags REAL,
    average_favorites REAL, average_retweet REAL, average_reply REAL,
    predicted_label TEXT NOT NULL,
    confidence      REAL NOT NULL,
    detected_on     TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    get_db().executescript(SCHEMA)
    get_db().commit()


def query(sql, args=(), one=False):
    cur = get_db().execute(sql, args)
    rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def execute(sql, args=()):
    db = get_db()
    cur = db.execute(sql, args)
    db.commit()
    return cur.lastrowid
