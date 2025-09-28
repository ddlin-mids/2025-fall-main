#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <input.ipynb> <output.ipynb>"
  exit 1
fi

in_nb="$1"
out_nb="$2"

if [[ ! -f "$in_nb" ]]; then
  echo "Error: input file not found: $in_nb" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq not found in PATH." >&2
  exit 1
fi

# temp file in same dir as output for atomic-ish replace (and to keep permissions sane)
out_dir="$(dirname "$out_nb")"
tmp_nb="$(mktemp "${out_dir%/}/.nbsanitize.XXXXXX")"

# The jq program that removes widget state globally, per-cell, and in outputs.
jq_program=$(cat <<'JQ'
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
)

# Run the sanitizer
jq "$jq_program" "$in_nb" > "$tmp_nb"

# Move into place
mv -f "$tmp_nb" "$out_nb"

echo "Sanitized notebook written to: $out_nb"
