#!/usr/bin/env bash
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scp barney:~/FYS-3941/data/test-results.npz "${SCRIPT_PATH}/../../$1"
