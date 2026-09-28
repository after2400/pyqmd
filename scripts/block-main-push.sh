#!/usr/bin/env bash
# pre-push hook: refuse direct pushes to main. Open a PR from a branch instead.
# Bypass deliberately with: git push --no-verify
#
# Run under pre-commit's pre-push integration, which exposes the ref being
# pushed to via PRE_COMMIT_REMOTE_BRANCH rather than forwarding git's raw
# pre-push stdin (local_ref local_sha remote_ref remote_sha) to system hooks.
# Fall back to parsing stdin so the script also works as a plain git hook.
set -euo pipefail

remote_branch="${PRE_COMMIT_REMOTE_BRANCH:-}"

if [[ -z "$remote_branch" ]]; then
    while read -r _local_ref _local_sha remote_ref _remote_sha; do
        if [[ "$remote_ref" == "refs/heads/main" ]]; then
            remote_branch="refs/heads/main"
            break
        fi
    done
fi

if [[ "$remote_branch" == "refs/heads/main" || "$remote_branch" == "main" ]]; then
    echo "error: direct pushes to 'main' are blocked -- push a branch and open a PR instead." >&2
    echo "  (deliberate bypass: git push --no-verify)" >&2
    exit 1
fi

exit 0
