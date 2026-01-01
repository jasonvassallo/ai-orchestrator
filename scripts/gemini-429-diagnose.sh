#!/usr/bin/env bash
set -u
set -o pipefail

log_path="${1:-$HOME/.gemini/debug.log}"

if [[ ! -f "$log_path" ]]; then
  echo "gemini-429-diagnose: debug log not found: $log_path" >&2
  echo "Hint: GEMINI_DEBUG_LOG_FILE=~/.gemini/debug.log gemini --debug ..." >&2
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  match_cmd=(rg -n)
  extract_cmd=(rg -o)
else
  match_cmd=(grep -n -E)
  extract_cmd=(grep -E -o)
fi

extract_retry_delay_seconds() {
  local raw num unit
  raw="$("${extract_cmd[@]}" 'Please retry in [0-9.]+(ms|s)|Suggested retry after [0-9.]+s|retryDelay\"[[:space:]]*:[[:space:]]*\"[0-9.]+s\"' "$log_path" | head -n1 || true)"
  if [[ -z "$raw" ]]; then
    return 1
  fi
  num="$(echo "$raw" | sed -E 's/.*([0-9.]+)(ms|s).*/\\1/')"
  unit="$(echo "$raw" | sed -E 's/.*([0-9.]+)(ms|s).*/\\2/')"
  if [[ "$unit" == "ms" ]]; then
    awk -v n="$num" 'BEGIN { printf "%.3f", n/1000 }'
  else
    echo "$num"
  fi
}

echo "gemini-429-diagnose: $log_path"
echo
echo "Matches (first 50 lines):"
"${match_cmd[@]}" 'QuotaFailure|ErrorInfo|RetryInfo|quota_limit|quotaId|retryDelay|PerDay|PerMinute|QUOTA_EXHAUSTED|RATE_LIMIT_EXCEEDED|Resource exhausted|Too Many Requests|429' "$log_path" | head -n 50 || true
echo

retry_delay=""
if retry_delay="$(extract_retry_delay_seconds)"; then
  echo "Parsed retry delay: ${retry_delay}s"
else
  echo "Parsed retry delay: (none found)"
fi

has_daily=false
has_per_min=false
has_quota_exhausted=false

"${match_cmd[@]}" 'PerDay|Daily' "$log_path" >/dev/null 2>&1 && has_daily=true
"${match_cmd[@]}" 'PerMinute|PerSecond|RATE_LIMIT_EXCEEDED' "$log_path" >/dev/null 2>&1 && has_per_min=true
"${match_cmd[@]}" 'QUOTA_EXHAUSTED' "$log_path" >/dev/null 2>&1 && has_quota_exhausted=true

echo
echo "Diagnosis:"
if [[ "$has_daily" == "true" ]]; then
  echo "- Likely daily quota exhaustion. Wait for reset or request a quota increase."
elif [[ "$has_per_min" == "true" ]]; then
  echo "- Likely short-term rate limit. Backoff and retry with the suggested delay."
elif [[ -n "$retry_delay" ]]; then
  if awk -v d="$retry_delay" 'BEGIN { exit (d > 120) ? 0 : 1 }'; then
    echo "- Long retry delay (>120s). Treat as capacity or longer-term quota."
  else
    echo "- Short retry delay (<=120s). Treat as rate limiting."
  fi
elif [[ "$has_quota_exhausted" == "true" ]]; then
  echo "- QUOTA_EXHAUSTED reported. Check Vertex AI quotas for the project/model."
else
  echo "- No structured quota details found. This often means capacity exhaustion on preview/global."
fi
