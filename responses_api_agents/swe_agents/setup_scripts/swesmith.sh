#!/bin/bash
set -e
set -x

# Variables
setup_dir=$SETUP_DIR
uv_dir=$UV_DIR
python_dir=$PYTHON_DIR
swesmith_dir=$SWESMITH_DIR
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

# Clone SWE-smith
if [ ! -d "$swesmith_dir/.git" ]; then
    echo "Cloning SWE-smith..."
    # Clean up any partial clone
    rm -rf "$swesmith_dir"
    git clone $eval_harness_repo $swesmith_dir
else
    echo "SWE-smith already cloned at $swesmith_dir"
fi

cd $swesmith_dir
echo "Checking out $eval_harness_commit..."
git checkout $eval_harness_commit

echo "Installing Python 3.11 to portable location..."
uv python install 3.11

echo "Python installations:"
uv python list

echo "Creating virtual environment with uv..."
rm -rf venv
uv venv --python 3.11 venv

# We only ever use swesmith.profiles.registry (get_from_inst/get_test_cmd/log_parser)
# and swesmith.harness.grading.get_eval_report -- both confirmed Docker-free, pure
# git/subprocess/string logic (see SWESmithDatasetProcessor's own comment for why: the
# package's own swesmith.harness.eval module launches a NEW Docker container internally via
# docker.from_env(), which doesn't exist on this cluster). The `validate` extra is SWE-smith's
# own "no sglang/litellm/openai/docker" minimal install group -- but swesmith/profiles/base.py
# still does an unconditional `import docker` at module level regardless of which extra was
# used to install it, so `docker` (the lightweight PyPI client library, NOT a running daemon --
# we never call any method that actually connects to one) is added explicitly alongside it.
echo "Installing SWE-smith (validate extra + docker, editable) in isolated venv..."
uv pip install -p $swesmith_dir/venv/bin/python -e ".[validate]" docker --no-cache

echo "Verifying installation..."
$swesmith_dir/venv/bin/python -c "from swesmith.profiles import registry; from swesmith.harness.grading import get_eval_report; print('✓ swesmith installed successfully, registry has', len(registry.keys()), 'repos')"

if [ -d venv ] && [ -f venv/bin/python ]; then
    echo "✓ venv created at $(pwd)/venv"
    echo "✓ Python version: $(venv/bin/python --version)"
else
    echo "✗ ERROR: venv was not created properly!"
    exit 1
fi

echo "SWE-smith setup complete!"
