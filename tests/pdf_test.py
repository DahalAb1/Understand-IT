import os, sys, time, subprocess

# ── 1. make project root importable ────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import utils   # import now resolves
# (no need to import app.main here; Streamlit will do it)

PDF_FILE = 'complex_test.pdf'          # located inside tests/

# ── 2. tiny regression check on utils workflow ────────────────────────────────
def _run_utils_pipeline():
    raw_text = utils.extract_text_from_pdf(PDF_FILE)
    chunks   = utils.chunk_text(raw_text)
    utils.reword_text(chunks)
    print("✅ utils pipeline ran without error")

# ── 3. helper to launch & kill Streamlit cleanly ───────────────────────────────
def _launch_streamlit(seconds: int = 10):
    # absolute path to app/main.py
    main_py = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'app', 'main.py'))

    proc = subprocess.Popen(
        ['streamlit', 'run', main_py],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid          # new process group → easy to kill
    )

    print(f"✅ Streamlit started (PID {proc.pid}) – running {seconds}s …")
    time.sleep(seconds)               # ← whatever window you need
    os.killpg(os.getpgid(proc.pid), 15)  # 15 = SIGTERM
    print("🛑 Streamlit terminated")

# ── 4. public test entrypoint (pytest will auto‑discover) ──────────────────────
def test_workflow():
    _run_utils_pipeline()
    _launch_streamlit(10)
