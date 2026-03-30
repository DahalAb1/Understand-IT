import sqlite3


class SqliteCacheAdapter:
    def __init__(self, db_path: str = "cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)"
            )

    def get(self, key: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, value),
            )
