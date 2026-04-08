"""Configure test paths so pytest doesn't trigger the package's __init__.py."""
import sys
from pathlib import Path

# Add the nodes directory directly to sys.path so tests can import
# nodes.minimax_tts_node without going through the package __init__.py
_repo_root = Path(__file__).parent.parent
_nodes_parent = _repo_root  # nodes/ lives at repo_root/nodes/
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
