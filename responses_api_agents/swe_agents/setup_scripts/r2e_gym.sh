#!/bin/bash
set -e
set -x

# Variables
setup_dir=$SETUP_DIR
uv_dir=$UV_DIR
python_dir=$PYTHON_DIR
r2e_gym_dir=$R2E_GYM_DIR
eval_harness_repo=$EVAL_HARNESS_REPO
eval_harness_commit=$EVAL_HARNESS_COMMIT

cd $setup_dir

export UV_INSTALL_DIR="$uv_dir"
# [FIXED 2026-10-13] uv's installer modifies the invoking user's shell rc file
# (~/.bashrc/.zshrc) by default to source "$UV_INSTALL_DIR/env" on every new shell --
# confirmed this had been silently happening on every run of this setup script, leaving
# stale `source ".../swe_<dataset>_setup/uv/env"` lines in a real user's .bashrc pointing at
# this project's own per-dataset scratch directories. Completely unnecessary: this script
# already does its own `export PATH="$uv_dir/bin:$PATH"` below, scoped to just this process,
# which is all any caller of this script needs. UV_NO_MODIFY_PATH=1 is uv's documented
# opt-out (confirmed by reading install.sh directly) -- never touch global shell state from
# a project setup script.
export UV_NO_MODIFY_PATH=1
export UV_PYTHON_INSTALL_DIR="$python_dir"
if [ ! -f "$uv_dir/bin/uv" ]; then
    echo "Installing uv to $uv_dir..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv already installed at $uv_dir"
fi

export PATH="$uv_dir/bin:$PATH"
echo "Verifying uv installation..."
which uv
uv --version

# Clone R2E-Gym
if [ ! -d "$r2e_gym_dir/.git" ]; then
    echo "Cloning R2E-Gym..."
    # Clean up any partial clone
    rm -rf "$r2e_gym_dir"
    git clone $eval_harness_repo $r2e_gym_dir
else
    echo "R2E-Gym already cloned at $r2e_gym_dir"
fi

cd $r2e_gym_dir
echo "Checking out $eval_harness_commit..."
git checkout $eval_harness_commit

echo "Installing Python 3.12 to portable location..."
uv python install 3.12

# [CUSTOM-R2EGYM-EVAL-VENV-PYTHON-3921 2026-10-15] R2E-Gym's own per-instance eval images (see
# the big relink comment in app.py's _copy_out_writable_paths, CUSTOM-R2EGYM-EVAL-VENV-PYTHON)
# bake `.venv/bin/python` as an absolute symlink to one of a small, fixed set of interpreter
# versions under the image-build-time-only /root: 3.9.21 for aiohttp/datalad/tornado, 
# 3.7.9 for coveragepy/pandas/numpy. That relink logic resolves the real interpreter from 
# THIS directory ($python_dir, shared via UV_PYTHON_INSTALL_DIR above). Installing it 
# explicitly here.
echo "Installing Python 3.9.21 for R2E-Gym eval-venv relinking (aiohttp/datalad/tornado)..."
uv python install 3.9.21

# [CUSTOM-R2EGYM-EVAL-VENV-PYTHON-379 2026-10-15] Python 3.7.9 (needed for coveragepy/pandas/
# numpy's eval venvs, see above) is EOL with no python-build-standalone release at all --
# confirmed `uv python install 3.7.9` fails outright ("No download found"). conda-forge DOES
# still publish a real linux-64 build (`python-3.7.9-hffdb5ce_100_cpython`), and it's fully
# self-contained 
# (RPATH `$ORIGIN/../lib`, no LD_LIBRARY_PATH needed, all core extensions incl. ssl/sqlite3/_ctypes
# present, works correctly as a venv base interpreter with cache_tag `cpython-37` matching real cp37 wheels)
# Using micromamba (a single static binary, no toolchain) rather than bootstrapping a full
# miniforge just to fetch one package.
py379_dir="$python_dir/cpython-3.7.9-linux-x86_64-gnu"
if [ -x "$py379_dir/bin/python3.7" ]; then
    echo "Python 3.7.9 (conda-forge) already installed at $py379_dir"
else
    echo "Installing Python 3.7.9 (conda-forge, via micromamba) for R2E-Gym eval-venv relinking (coveragepy/pandas/numpy)..."
    micromamba_tmp="$setup_dir/micromamba_bootstrap"
    mkdir -p "$micromamba_tmp"
    if [ ! -x "$micromamba_tmp/bin/micromamba" ]; then
        curl -LsS "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xj -C "$micromamba_tmp" bin/micromamba
    fi
    rm -rf "$py379_dir"
    MAMBA_ROOT_PREFIX="$setup_dir/micromamba_root" "$micromamba_tmp/bin/micromamba" create -p "$py379_dir" -c conda-forge python=3.7.9 -y
    "$py379_dir/bin/python3.7" -c "import ssl, sqlite3, _ctypes" || {
        echo "ERROR: conda-forge Python 3.7.9 install at $py379_dir failed a basic import check" >&2
        exit 1
    }
    # micromamba_root is just the download/package cache -- the installed env at $py379_dir is
    # fully self-contained (confirmed via RPATH $ORIGIN/../lib), so the cache has no further
    # value after a successful create.
    rm -rf "$setup_dir/micromamba_root"
fi

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.12 venv

echo "Installing R2E-Gym in editable mode..."
uv pip install -p $r2e_gym_dir/venv/bin/python -e . --no-cache

echo "Verifying installation..."
$r2e_gym_dir/venv/bin/python -c "import r2egym; print('✓ r2egym installed successfully')"

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "R2E-Gym setup complete!"
