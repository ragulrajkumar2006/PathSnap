#!/bin/zsh
set -eu
cd "${0:A:h}"
iface=$(route -n get default 2>/dev/null | awk '/interface:/ {print $2; exit}')
iface=${iface:-en0}
ipv4=$(ipconfig getifaddr "$iface" 2>/dev/null || true)
ipv6=$(ifconfig "$iface" | awk '$1=="inet6" && $2 ~ /^[23]/ {print $2; exit}')
if [[ -n "$ipv6" && ( -z "$ipv4" || "$ipv4" == 192.0.0.* ) ]]; then
    bind_host='::'
    url_host="[$ipv6]"
else
    bind_host=0.0.0.0
    if [[ -n "$ipv4" && "$ipv4" != 192.0.0.* ]]; then
        url_host="$ipv4"
    else
        url_host="$(scutil --get LocalHostName).local"
    fi
fi
echo "Laptop 2 browser check: http://${url_host}:8000/trips"
echo "Laptop 2 UI API URL:    http://${url_host}:8000"
echo "Wait for Uvicorn running, then leave this window open."
exec .venv/bin/python -m uvicorn mac1.api.main:app --host "$bind_host" --port 8000
