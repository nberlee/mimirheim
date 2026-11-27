"""Make the ``dhw_benchmark`` package importable from ``scripts/``.

The benchmark harness is a local tool under ``scripts/``, not an installed
package, so its tests add that directory to the import path.
"""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
