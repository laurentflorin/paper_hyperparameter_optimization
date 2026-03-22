#!/usr/bin/env bash

set -euo pipefail

target_dir="${HOME}/.local/bin"
target_path="${target_dir}/codex"

mkdir -p "${target_dir}"

cat > "${target_path}" <<'EOF'
#!/usr/bin/env bash

set -euo pipefail

extensions_dir="${HOME}/.vscode-remote/extensions"

case "$(uname -m)" in
  x86_64|amd64)
    platform_dir="linux-x86_64"
    ;;
  aarch64|arm64)
    platform_dir="linux-aarch64"
    ;;
  *)
    echo "Unsupported architecture for bundled Codex CLI: $(uname -m)" >&2
    exit 1
    ;;
esac

latest_extension="$(
  find "${extensions_dir}" -maxdepth 1 -mindepth 1 -type d -name 'openai.chatgpt-*-linux-*' \
    | sort -V \
    | tail -n 1
)"

if [[ -z "${latest_extension}" ]]; then
  echo "Could not find the OpenAI Codex VS Code extension under ${extensions_dir}." >&2
  echo "Install or reopen the Codex extension in VS Code, then try again." >&2
  exit 1
fi

bundled_codex="${latest_extension}/bin/${platform_dir}/codex"

if [[ ! -x "${bundled_codex}" ]]; then
  echo "Found ${latest_extension}, but the bundled Codex binary is missing at ${bundled_codex}." >&2
  exit 1
fi

exec "${bundled_codex}" "$@"
EOF

chmod 755 "${target_path}"

echo "Installed Codex launcher at ${target_path}"
echo "Try: codex --version"
echo "Resume last session with: codex resume --last"
