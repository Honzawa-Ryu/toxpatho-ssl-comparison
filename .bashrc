# Load scripts in .bashrc.d
BASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "${BASH_DIR}/.bashrc.d" ]; then
    for f in "${BASH_DIR}/.bashrc.d"/*.sh; do
        [ -r "$f" ] && source "$f"
    done
    unset f
fi
unset BASH_DIR

# jq は Slack通知（scripts/notify_slack.sh）のペイロード構築にのみ使う。
# ここで `exit` してはいけない: このファイルは ~/.bashrc から source される想定であり、
# 非対話シェル（scp/sftp/rsync のセッションを含む）まで巻き添えで落ちる。
# 同じ理由で、成功時も stdout には何も出さない（余計な stdout は scp/sftp を壊す）。
if _is_interactive_shell 2>/dev/null && ! command -v jq >/dev/null 2>&1; then
    echo "⚠️  jq が見つかりません。Slack通知が動作しません。" >&2
fi