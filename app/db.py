import os
import sqlite3
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "real_estate.db"
SQL_DIR = BASE_DIR / "sql"

DATABASE_URL = os.getenv("DATABASE_URL", "")
_active_backend = "sqlite" if not DATABASE_URL.startswith("postgresql") else "postgres"


def get_active_backend() -> str:
    return _active_backend


def _use_postgres() -> bool:
    return _active_backend == "postgres"


def get_engine() -> Engine | None:
    if _use_postgres():
        return create_engine(DATABASE_URL)
    return None


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _run_sql_file(engine: Engine, sql_path: Path) -> None:
    sql = sql_path.read_text()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))


def initialize_database(force: bool = False) -> None:
    global _active_backend

    if DATABASE_URL.startswith("postgresql"):
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            _active_backend = "postgres"
            if force or not _postgres_has_data(engine):
                _run_sql_file(engine, SQL_DIR / "create_tables.sql")
                _run_sql_file(engine, SQL_DIR / "seed_data.sql")
            return
        except OperationalError:
            _active_backend = "sqlite"
            print("Postgres unavailable — falling back to SQLite at data/real_estate.db")

    if DB_PATH.exists() and not force:
        return

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS financials")
    cursor.execute("DROP TABLE IF EXISTS properties")

    cursor.execute(
        """
        CREATE TABLE properties (
            property_id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT NOT NULL,
            metro_area TEXT NOT NULL,
            sq_footage INTEGER NOT NULL,
            property_type TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE financials (
            financial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER,
            revenue REAL,
            net_income REAL,
            expenses REAL,
            FOREIGN KEY(property_id) REFERENCES properties(property_id)
        )
        """
    )

    seed_sql = (SQL_DIR / "seed_data.sql").read_text()
    for statement in seed_sql.split(";"):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)

    conn.commit()
    conn.close()


def _postgres_has_data(engine: Engine) -> bool:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM properties"))
        return result.scalar() > 0


def _read_query(query: str) -> pd.DataFrame:
    engine = get_engine()
    if engine is not None:
        return pd.read_sql_query(query, engine)

    conn = get_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_property_financials() -> pd.DataFrame:
    query = """
        SELECT
            p.property_id,
            p.address,
            p.metro_area,
            p.sq_footage,
            p.property_type,
            f.revenue,
            f.net_income,
            f.expenses
        FROM properties p
        JOIN financials f ON p.property_id = f.property_id
        ORDER BY p.property_id
    """
    return _read_query(query)


def _normalize_metro(metro_area: str | None) -> str | None:
    if not metro_area or metro_area == "All":
        return metro_area
    normalized = metro_area.strip()
    for suffix in (" region", " metro", " metro area", " area"):
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    return normalized


def filter_properties(metro_area=None, property_type=None) -> pd.DataFrame:
    df = get_property_financials()

    metro_area = _normalize_metro(metro_area)
    if metro_area and metro_area != "All":
        df = df[df["metro_area"].str.lower() == metro_area.lower()]

    if property_type and property_type != "All":
        df = df[df["property_type"].str.lower() == property_type.lower()]

    return df


def get_portfolio_summary() -> dict:
    df = get_property_financials()
    return {
        "total_properties": int(df["property_id"].nunique()),
        "total_revenue": float(df["revenue"].sum()),
        "total_net_income": float(df["net_income"].sum()),
        "total_expenses": float(df["expenses"].sum()),
    }
