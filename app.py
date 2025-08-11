# Root-level entry for Render deployment wrapping the existing module in subfolder.
import os
from pathlib import Path
import importlib.util

SUBDIR = Path(__file__).parent / 'heart disease predictor' / 'app.py'
if not SUBDIR.exists():
    raise RuntimeError(f"Cannot locate app module at {SUBDIR}")

spec = importlib.util.spec_from_file_location('heart_app', SUBDIR)
mod = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(mod)  # type: ignore
app = mod.app  # Flask instance

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
