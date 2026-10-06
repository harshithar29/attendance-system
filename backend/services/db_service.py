"""
backend/services/db_service.py

Handles all database connectivity for the application.
Supports MySQL with an automatic fallback to local SQLite (database/attendance.db)
so that the application can run in the cloud (e.g. Render) without requiring an
external database server, while still seamlessly connecting to MySQL whenever available.
"""

import logging
import os
import re
import sqlite3
from datetime import datetime

import mysql.connector
from mysql.connector import pooling

from config.config import Config

logger = logging.getLogger(__name__)

_pool = None
_use_sqlite = False
_sqlite_path = os.path.join(Config.BASE_DIR, "database", "attendance.db")


def _get_sqlite_conn():
    conn = sqlite3.connect(_sqlite_path)
    conn.row_factory = sqlite3.Row
    # Register MySQL-compatible SQL functions in SQLite
    conn.create_function("NOW", 0, lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    conn.create_function("CURDATE", 0, lambda: datetime.now().strftime("%Y-%m-%d"))
    conn.create_function("DATE", 1, lambda val: str(val)[:10] if val else None)
    conn.create_function("IFNULL", 2, lambda a, b: b if a is None else a)
    conn.create_function("LAST_DAY", 1, lambda val: str(val)[:7] + "-28" if val else None)
    return conn


def init_db_pool():
    """
    Creates a MySQL connection pool once at application startup.
    If MySQL cannot be reached (e.g. when hosted on Render or cloud),
    it automatically falls back to the embedded SQLite database.
    """
    global _pool, _use_sqlite
    if _pool is not None or _use_sqlite:
        return _pool

    if os.environ.get("USE_SQLITE", "false").lower() in ("true", "1") or not Config.DB_HOST:
        logger.info("Using SQLite database at %s", _sqlite_path)
        _use_sqlite = True
        return None

    try:
        _pool = pooling.MySQLConnectionPool(
            pool_name="attendance_pool",
            pool_size=5,
            host=Config.DB_HOST,
            port=Config.DB_PORT,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
            connection_timeout=5,
        )
        logger.info("MySQL connection pool created successfully.")
    except Exception as err:
        logger.warning(
            "MySQL connection failed (%s). Falling back to SQLite database at %s",
            err,
            _sqlite_path,
        )
        _use_sqlite = True
    return _pool


def get_connection():
    """Returns a single connection borrowed from the pool or an SQLite connection."""
    global _pool, _use_sqlite
    if _use_sqlite:
        return _get_sqlite_conn()
    if _pool is None:
        init_db_pool()
        if _use_sqlite:
            return _get_sqlite_conn()
    try:
        return _pool.get_connection()
    except Exception as err:
        logger.warning("Failed to borrow MySQL connection: %s. Using SQLite fallback.", err)
        _use_sqlite = True
        return _get_sqlite_conn()


def run_query(query, params=None, fetch_one=False, fetch_all=False, commit=False):
    """
    Helper that wraps: open connection -> get cursor -> execute -> fetch/commit -> close.
    Works seamlessly across both MySQL and SQLite.
    """
    global _use_sqlite
    if _use_sqlite or _pool is None:
        init_db_pool()

    if _use_sqlite:
        conn = _get_sqlite_conn()
        cursor = conn.cursor()
        result = None
        try:
            # Replace %s with ? for SQLite parameters
            # Preserve string literals containing %
            converted_query = re.sub(r"%s", "?", query)
            cursor.execute(converted_query, params or ())
            if commit:
                conn.commit()
                result = cursor.lastrowid
            elif fetch_one:
                row = cursor.fetchone()
                result = dict(row) if row else None
            elif fetch_all:
                rows = cursor.fetchall()
                result = [dict(r) for r in rows]
        finally:
            cursor.close()
            conn.close()
        return result

    # MySQL path
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    result = None
    try:
        cursor.execute(query, params or ())
        if commit:
            conn.commit()
            result = cursor.lastrowid
        elif fetch_one:
            result = cursor.fetchone()
        elif fetch_all:
            result = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()
    return result
