# SimplifierService — the brain. Implements SimplifierPort.
# Receives PdfReaderPort, ModelPort, CachePort via constructor (dependency injection).
# Orchestrates: extract → chunk → cache check → model → cache write → return result.
# Zero imports from sqlite3, transformers, fastapi, or pypdf.
