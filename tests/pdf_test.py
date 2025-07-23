import os, sys, time, subprocess

# to make project root importable 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import utils  

PDF_FILE = 'complex_test.pdf'


def _run_utils_pipeline():
    raw_text = utils.extract_text_from_pdf(PDF_FILE)
    chunks   = utils.chunk_text(raw_text)
    utils.reword_text(chunks)
    print("✅ utils pipeline ran without error")

# to launch & kill Streamlit cleanly
def _launch_streamlit(seconds: int = 10):
    # path to app/main.py
    main_py = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'app', 'main.py'))

    proc = subprocess.Popen(
        ['streamlit', 'run', main_py],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )

    print(f"✅ Streamlit started (PID {proc.pid}) – running {seconds}s …")
    time.sleep(seconds)
    os.killpg(os.getpgid(proc.pid), 15)
    print("🛑 Streamlit terminated")

# ── 4. public test entrypoint (pytest will auto‑discover)
def test_workflow():
    _run_utils_pipeline()
    _launch_streamlit(10)
