#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  nb-sanitize-widgets.sh --target-nb <input.ipynb> --output-nb <output.ipynb>

Description:
  Removes Jupyter widget state that breaks GitHub rendering while preserving
  code, markdown, and non-widget outputs (e.g., text/plain, image/png).

Options:
  --target-nb <PATH>   Path to the input notebook (.ipynb)
  --output-nb <PATH>   Path to write the sanitized notebook
  --help               Show this help message and exit

Examples:
  nb-sanitize-widgets.sh --target-nb Text_classification_CNN.ipynb \
                         --output-nb Text_classification_CNN_clean.ipynb

  # In-place sanitize
  nb-sanitize-widgets.sh --target-nb Text_classification_CNN.ipynb \
                         --output-nb Text_classification_CNN.ipynb
EOF
}

# --- Parse args ---
TARGET_NB=""
OUTPUT_NB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help)
      show_help; exit 0;;
    --target-nb)
      [[ $# -ge 2 ]] || { echo "Error: --target-nb requires a value" >&2; exit 1; }
      TARGET_NB="$2"; shift 2;;
    --output-nb)
      [[ $# -ge 2 ]] || { echo "Error: --output-nb requires a value" >&2; exit 1; }
      OUTPUT_NB="$2"; shift 2;;
    *)
      echo "Error: Unknown option: $1" >&2
      echo "Run with --help for usage." >&2
      exit 1;;
  esac
done

# --- Validate ---
[[ -n "${TARGET_NB}" ]] || { echo "Error: --target-nb is required" >&2; exit 1; }
[[ -n "${OUTPUT_NB}" ]] || { echo "Error: --output-nb is required" >&2; exit 1; }
[[ -f "${TARGET_NB}" ]] || { echo "Error: input file not found: ${TARGET_NB}" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "Error: jq not found in PATH" >&2; exit 1; }

# Temp file in output dir for atomic-ish write
OUT_DIR="$(dirname "${OUTPUT_NB}")"
mkdir -p "${OUT_DIR}"
TMP_NB="$(mktemp "${OUT_DIR%/}/.nbsanitize.XXXXXX")"

# --- jq program: strip widget state globally, per-cell, and in outputs ---
read -r -d '' JQ_PROGRAM <<'JQ' || true
del(.metadata.widgets)
| .cells |= map(
    (.metadata? |= del(.widgets))
    | (if ((.outputs? | type) == "array") then
         .outputs |= map(
           if ((.data? | type) == "object") then
             (.data |= (del(."application/vnd.jupyter.widget-view+json")
                        | del(."application/vnd.jupyter.widget-state+json")))
             | (if ((.data | keys | length) == 0) then del(.data) else . end)
           else
             .
           end
         )
       else
         .
       end)
  )
JQ

# --- Run sanitizer ---
jq "${JQ_PROGRAM}" "${TARGET_NB}" > "${TMP_NB}"

# --- Move into place ---
mv -f "${TMP_NB}" "${OUTPUT_NB}"

echo "Sanitized notebook written to: ${OUTPUT_NB}"
