# $0 Cloud Integration & 8GB Laptop Guide

This guide explains how to connect the platform to free cloud services (`$0`, no credit card required) and keep Docker light on an 8 GB RAM Windows 10 laptop.

---

## 1. Connect Neon Cloud PostgreSQL (`$0`, Free Tier)

1. Create a free project at **Neon** (`neon.tech` — includes `1 GB` storage and `100 CU-hours/month` on the Free plan).
2. Copy the PostgreSQL connection string from the Neon dashboard (it looks like `postgresql://USER:PASSWORD@ep-xxxx.neon.tech/neondb?sslmode=require`).
3. Create a `.env` file in the project root from `.env.example`:
   ```bash
   cp .env.example .env
   ```
4. Paste your Neon URL into `DATABASE_URL`:
   ```env
   DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx.neon.tech/neondb?sslmode=require
   ```
   *(The application automatically converts `postgresql://` to `postgresql+psycopg://` at startup.)*
5. Start the API container or local Uvicorn server and verify:
   ```bash
   curl http://127.0.0.1:8000/ready
   # Expected: {"status":"ready","model_version":"transformer-onnx-int8-v1","storage_backend":"postgresql"}
   ```

---

## 2. Optional: Grafana Cloud Free Tier

Grafana Cloud's Free plan includes `10,000` Prometheus metrics series. To push metrics from your local Prometheus container to Grafana Cloud:
1. Open your Grafana Cloud Portal -> **Prometheus** -> **Details** to get your `remote_write` URL, Username, and API Token.
2. Add a `remote_write` block at the bottom of `monitoring/prometheus/prometheus.yml`:
   ```yaml
   remote_write:
     - url: https://prometheus-prod-xx.grafana.net/api/prom/push
       basic_auth:
         username: "<YOUR_INSTANCE_ID>"
         password: "<YOUR_GRAFANA_CLOUD_TOKEN>"
   ```

---

## 3. Windows 10 (8 GB RAM + HDD) Optimization Tip

Docker Desktop on Windows uses WSL2, which can claim up to half of system RAM by default. To keep your laptop responsive:

1. Create or edit `C:\Users\<YourUser>\.wslconfig`:
   ```ini
   [wsl2]
   memory=3GB
   processors=2
   swap=2GB
   ```
2. Restart WSL2 from PowerShell:
   ```powershell
   wsl --shutdown
   ```
3. Notice that `Dockerfile` installs **only** `requirements-serve.txt` (`onnxruntime`, `fastapi`, `sqlalchemy`, `prometheus-client`) and does **not** install PyTorch into the container, keeping builds fast and container memory under `384 MiB`.
