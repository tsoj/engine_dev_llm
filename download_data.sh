#!/usr/bin/env bash
# Export every channel of the given Discord servers as JSON into data/discord_json_data/.
#
# Usage: DISCORD_TOKEN=... ./download_data.sh /path/to/DiscordChatExporter.Cli <server_id>...
#
# The token is read from the environment so it doesn't end up in shell history,
# `ps` output, or this script's log.
set -euo pipefail

if [[ $# -lt 2 || -z "${DISCORD_TOKEN:-}" ]]; then
  echo "Usage: DISCORD_TOKEN=... $0 /path/to/DiscordChatExporter.Cli <server_id>..." >&2
  exit 1
fi

exe=$1
shift

for server_id in "$@"; do
  echo "Exporting server $server_id"
  "$exe" exportguild -t "$DISCORD_TOKEN" -g "$server_id" -f Json \
    -o "./data/discord_json_data/[%G|%C][%g|%c].json"
done
