# Outbound adapter. Implements ModelPort.
# All transformers/HuggingFace code lives here.
# Lazy loads FLAN-T5-Large on first call.
# Exposes max_input_length = 512 so the domain knows how to chunk.
# Runs model.generate() and returns decoded text.
