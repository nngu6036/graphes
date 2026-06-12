from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from evaluate_grapher_metrics import main
if __name__ == '__main__':
    sys.argv.extend(['--model', 'dhvae']) if '--model' not in sys.argv else None
    main()
