"""Execute and save the EDA notebook with the current Python interpreter."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / 'tmp' / 'jupyter'
runtime.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(ROOT / 'tmp' / 'matplotlib')

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = ROOT / 'notebooks' / '01_eda.ipynb'
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=240,
                        resources={'metadata': {'path': str(ROOT)}})
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f'Executed {sum(c.cell_type == "code" for c in notebook.cells)} code cells: {path}')
