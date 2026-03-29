import sqlite3 
import hashlib 

con = sqlite3.connect('cache.db',check_same_thread=False)
cursor = con.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS cache (
    hash TEXT PRIMARY KEY, 
    text TEXT
    )
"""
)
con.commit()


def _key(txt):
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()



#here, used ? and ,(_key(txt),)//passed tuple, to prevent sql injection
def get(txt) -> str|None:
    row = cursor.execute(
        "SELECT text FROM cache WHERE hash = ?",(_key(txt),)
    ).fetchone()

    #fetchone returns a tuple, row[0] extracts it. 
    return row[0] if row else None

#public API, writes to cache. 
def set_cache(txt, rewritten):
    con.execute(
        "REPLACE INTO cache VALUES(?,?)",(_key(txt),rewritten)
    )
    con.commit()


