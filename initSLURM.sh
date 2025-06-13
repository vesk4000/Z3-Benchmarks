#!/bin/sh
set -eu

exec "$@"

# Execute the given command as benchuser
# if [ "$(id -u)" = "0" ]; then
#   exec su benchuser -c "$(printf '%q ' "$@")"
# else
#   exec "$@"
# fi