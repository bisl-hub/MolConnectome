"""Import every module under scripts/, each in a fresh interpreter, as `python <script>` would."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHILD = (
    'import sys, pathlib, importlib.util; path = sys.argv[1]; '
    'sys.path.insert(0, str(pathlib.Path(path).parent)); '
    'spec = importlib.util.spec_from_file_location("module_under_test", path); '
    'spec.loader.exec_module(importlib.util.module_from_spec(spec))'
)

files = sorted(p for p in (ROOT / 'scripts').rglob('*.py') if '__pycache__' not in p.parts)
failed = []
for f in files:
    r = subprocess.run([sys.executable, '-B', '-c', CHILD, str(f)], capture_output=True, text=True)
    print(f'{"ok" if r.returncode == 0 else "FAIL":4}  {f.relative_to(ROOT).as_posix()}')
    if r.returncode != 0:
        print(r.stderr)
        failed.append(f)
print(f'{len(files) - len(failed)}/{len(files)} modules imported')
sys.exit(1 if failed else 0)
