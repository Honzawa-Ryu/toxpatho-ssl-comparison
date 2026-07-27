#!/bin/bash
#SBATCH --job-name=run_jupyter_server
#SBATCH --gres=gpu:0
#SBATCH --output=jupyter-server.out
#SBATCH --error=jupyter-server.out
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00

# 1. Dynamically allocate a free port using Python
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# 2. Define node info and generate a secure token
COMPUTE_NODE=$(hostname -s)
COMPUTE_SSH_PORT=49152
TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)

# 3. Print instructions for SSH tunneling to bypass UFW
echo "=================================================================="
echo "Jupyter Server starting on: ${COMPUTE_NODE}"
echo ""
echo "[STEP 1: Create SSH Tunnel (Auto-closing)]"
echo "Open a NEW terminal tab in VS Code (on the login node) and run:"
echo "ssh -N -L ${PORT}:localhost:${PORT} -p ${COMPUTE_SSH_PORT} ${COMPUTE_NODE}"
echo ""
echo "  * Keep that terminal open while working."
echo "  * If you get 'Address already in use', change the FIRST ${PORT} to a random number."
echo ""
echo "[STEP 2: Connect to VS Code]"
echo "Copy this URL into VS Code's 'Select Kernel' -> 'Existing Jupyter Server':"
echo "http://localhost:${PORT}/?token=${TOKEN}"
echo "=================================================================="

# 4. Start Jupyter binding to localhost (127.0.0.1) for maximum security
apptainer exec --nv ${SIF_PATH} bash -c "
    source .venv/bin/activate && \
    jupyter lab --ip=127.0.0.1 --port=$PORT --no-browser --allow-root --ServerApp.token='$TOKEN'
"