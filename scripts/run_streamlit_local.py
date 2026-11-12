"""Launch the local PCB app with the selected on-demand model backend."""
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    source = str(root / 'src')
    os.environ['PYTHONPATH'] = source + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path.insert(0, source)
    from circuitcare.vision.weight_paths import resolve_pcb_weights_root
    os.environ['CIRCUITCARE_PCB_WEIGHTS'] = str(resolve_pcb_weights_root(root / "apps/pcb-component-recognition"))
    backend = os.environ.setdefault('CIRCUITCARE_LLM_BACKEND', 'mlx' if sys.platform == 'darwin' else 'ollama')
    if backend == 'mlx':
        interpreter = Path(os.environ.get('CIRCUITCARE_MLX_PYTHON',
            str(Path.home() / '.cache/circuitcare/venvs/mlx/bin/python')))
        if not interpreter.is_file():
            raise SystemExit('MLX Python environment is missing. Set CIRCUITCARE_MLX_PYTHON.')
        os.environ['CIRCUITCARE_MLX_PYTHON'] = str(interpreter)
    os.execv(sys.executable, [sys.executable, '-m', 'streamlit', 'run',
        str(root / 'src/circuitcare/ui.py'), '--server.address', '127.0.0.1',
        '--server.port', '8765', '--server.headless', 'true'])


if __name__ == '__main__':
    main()
