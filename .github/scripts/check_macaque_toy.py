"""Check a macaque toy-data run (steps 1 and 3) against MACAQUE_PIPELINE.md section 6.

Usage: python check_macaque_toy.py <output_dir>/background_genes_00_transformation
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

run_dir = Path(sys.argv[1])
d = pd.read_csv(run_dir / 'directed_signaling_strength.csv')
z_bound = abs(scipy.stats.norm.ppf(0.5 / 1001))  # 3.290808, reachable

sheets = pd.read_excel(run_dir / 'weighted_connectivity.xlsx', sheet_name=None, index_col=0)
total = sheets['total'].to_numpy(float)
off_diagonal = total[~np.eye(len(total), dtype=bool)]

checks = {
    'step 1: 1890 rows x 14 columns': d.shape == (1890, 14),
    'step 1: no NaN': d.isna().sum().sum() == 0,
    'step 1: n_monkeys within 11-15': bool(d['n_monkeys'].between(11, 15).all()),
    'step 1: |mean_z| <= 3.290808': d['mean_z'].abs().max() <= z_bound + 1e-9,
    'step 1: 31 rows exactly on the bound': int(np.isclose(d['mean_z'].abs(), z_bound, rtol=0, atol=1e-9).sum()) == 31,
    'step 1: 291 rows with mean_z > 1.645': int((d['mean_z'] > 1.645).sum()) == 291,
    'step 1: 225 rows with mean_z < -1.645': int((d['mean_z'] < -1.645).sum()) == 225,
    'step 3: total + 9 weighted_<NT> sheets': len(sheets) == 10 and all(s.shape == (15, 15) for s in sheets.values()),
    'step 3: NaN only on the diagonal of total': bool(np.isnan(np.diag(total)).all() and not np.isnan(off_diagonal).any()),
    'step 3: total off-diagonal min 0.856': abs(off_diagonal.min() - 0.856) < 1e-3,
    'step 3: total off-diagonal max 3.126': abs(off_diagonal.max() - 3.126) < 1e-3,
}
for name, ok in checks.items():
    print(f'{"ok" if ok else "FAIL":4}  {name}')
sys.exit(0 if all(checks.values()) else 1)
