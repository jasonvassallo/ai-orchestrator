#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage: gemini-retry.sh [options] "<prompt>"

Options:
  -r, --resume SESSION        Session to resume (default: latest)
  -m, --model MODEL           Override model
      --max-attempts N        Max attempts (default: 6)
      --initial-delay SECS    Initial delay between attempts (default: 30)
      --max-delay SECS        Max delay between attempts (default: 600)
  -h, --help                  Show help

Env overrides:
  GEMINI_RETRY_RESUME
  GEMINI_RETRY_MODEL
  GEMINI_RETRY_MAX_ATTEMPTS
  GEMINI_RETRY_INITIAL_DELAY
  GEMINI_RETRY_MAX_DELAY
EOF
}

max_attempts="${GEMINI_RETRY_MAX_ATTEMPTS:-6}"
initial_delay="${GEMINI_RETRY_INITIAL_DELAY:-30}"
max_delay="${GEMINI_RETRY_MAX_DELAY:-600}"
resume="${GEMINI_RETRY_RESUME:-latest}"
model="${GEMINI_RETRY_MODEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--resume)
      resume="${2:-}"
      shift 2
      ;;
    -m|--model)
      model="${2:-}"
      shift 2
      ;;
    --max-attempts)
      max_attempts="${2:-}"
      shift 2
      ;;
    --initial-delay)
      initial_delay="${2:-}"
      shift 2
      ;;
    --max-delay)
      max_delay="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

prompt="$*"
model_args=()
if [[ -n "$model" ]]; then
  model_args=(--model "$model")
fi

if command -v rg >/dev/null 2>&1; then
  match_cmd=(rg -q)
  extract_cmd=(rg -o)
else
  match_cmd=(grep -q -E)
  extract_cmd=(grep -E -o)
fi

is_retryable() {
  local file="$1"
  "${match_cmd[@]}" 'Resource exhausted|Too Many Requests|Rate limit|QUOTA|Quota|quota|status[[:space:]]*429|code[[:space:]]*429|\\b429\\b' "$file"
}

extract_retry_delay_seconds() {
  local file="$1"
  local raw num unit
  raw="$("${extract_cmd[@]}" 'Please retry in [0-9.]+(ms|s)|Suggested retry after [0-9.]+s|retryDelay\"[[:space:]]*:[[:space:]]*\"[0-9.]+s\"' "$file" | head -n1 || true)"
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

attempt=1
delay="$initial_delay"

while (( attempt <= max_attempts )); do
  tmpfile="$(mktemp)"
  echo "gemini-retry: attempt ${attempt}/${max_attempts} (delay=${delay}s)"
  NO_COLOR=1 gemini "${model_args[@]}" --resume "$resume" "$prompt" 2>&1 | tee "$tmpfile"
  gemini_status=${PIPESTATUS[0]}

  if [[ $gemini_status -eq 0 ]]; then
    rm -f "$tmpfile"
    exit 0
  fi

  if ! is_retryable "$tmpfile"; then
    rm -f "$tmpfile"
    exit "$gemini_status"
  fi

  if retry_delay="$(extract_retry_delay_seconds "$tmpfile")"; then
    delay="$retry_delay"
  fi

  rm -f "$tmpfile"

  if (( attempt == max_attempts )); then
    exit "$gemini_status"
  fi

  sleep "$delay"
  delay=$(awk -v d="$delay" -v max="$max_delay" 'BEGIN { next = d * 2; if (next > max) next = max; printf "%.3f", next }')
  attempt=$((attempt + 1))
done

exit 1
