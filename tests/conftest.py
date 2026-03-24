import os
import sys
from pathlib import Path

# Enable MPS fallback so ops not yet implemented on Apple Silicon (e.g. CTC loss)
# transparently fall back to CPU.  Must be set before the first ``import torch``.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))
