"""Regenerate every golden fixture into its package's testdata/ directory."""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

for script in sorted(glob.glob(os.path.join(HERE, "generate_*_fixtures.py"))):
    print("==>", os.path.basename(script))
    subprocess.run([sys.executable, script], check=True)
