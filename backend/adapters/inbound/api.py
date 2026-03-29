# Inbound adapter. FastAPI router.
# One route: POST /simplify
# Receives PDF file from React, calls SimplifierPort.simplify(), returns JSON.
# Knows about HTTP. Knows nothing about the model, cache, or PDF parsing.
