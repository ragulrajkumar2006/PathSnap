# PathSnap UI (Laptop 2)

This folder is everything Laptop 2 needs. It holds no model, trip data or reference routes; it only talks to the API on Laptop 1.

## One-time setup

You need Python 3.11–3.13 (`python3 --version`).

```bash
cd laptop2_ui
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

1. Connect to the same Wi-Fi or hotspot as Laptop 1.
2. Get Laptop 1's IP (on Laptop 1: `ipconfig getifaddr en0`).
3. Check that Laptop 1's API is reachable by opening `http://<LAPTOP1_IP>:8000/trips` in a browser. It should list the trips.
4. Start the UI:

```bash
cd laptop2_ui
source .venv/bin/activate
streamlit run app.py -- --api http://<LAPTOP1_IP>:8000
```

The browser opens at `http://localhost:8501`. Pick a trip and click **Reconstruct route**.
You can also change the API URL in the sidebar while the UI is running.

## If it can't connect
- Make sure both laptops are on the same network. A phone hotspot is safest.
- Laptop 1 must have started the API with `--host 0.0.0.0`.
- On Laptop 1, go to System Settings → Network → Firewall and allow incoming connections for Python, or turn the firewall off for the demo.
