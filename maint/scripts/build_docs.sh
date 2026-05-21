#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

export UV_INDEX_URL="http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple"
export UV_EXTRA_INDEX_URL="https://repo.huaweicloud.com/ascend/repos/pypi"
export UV_INDEX_STRATEGY="unsafe-best-match"
export UV_INSECURE_HOST="cache-service.nginx-pypi-cache.svc.cluster.local"
export UV_HTTP_TIMEOUT=120
export UV_NO_CACHE=1
export UV_SYSTEM_PYTHON=1

python -m venv .venv
source .venv/bin/activate
python -m pip install uv
uv pip install -r requirements-test.txt
uv pip install -r docs/requirements.txt
uv pip install -e .

cd docs
make html

cp CNAME _build/html/
