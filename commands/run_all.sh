#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# All dataset commands use the finalized main generation profile:
# - generic: DCR tau=0.4 + B1to4 degree-preserving rewiring;
# - attributed: strict B1to4 same-bond rewiring on the fixed-topology model.
bash commands/community_small.sh
bash commands/ego_small.sh
bash commands/qm9.sh
bash commands/zinc.sh
