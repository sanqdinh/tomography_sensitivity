#!/usr/bin/env bash
# Start the local app and print a URL reachable from other devices on this LAN.
set -euo pipefail

port="${STREAMLIT_SERVER_PORT:-8501}"
lan_ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | awk 'NF && $1 !~ /^127\./ { print; exit }')"

if [[ -n "${lan_ip}" ]]; then
    printf 'LAN URL: http://%s:%s\n' "${lan_ip}" "${port}"
else
    printf 'Could not determine a LAN address; Streamlit will still listen on port %s.\n' "${port}" >&2
fi

exec streamlit run app.py "$@"
