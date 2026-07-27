# Load scripts in .bashrc.d
BASH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -d "${BASH_DIR}/.bashrc.d" ]; then
    for f in "${BASH_DIR}/.bashrc.d"/*.sh; do
        [ -r "$f" ] && source "$f"
    done
    unset f
fi
unset BASH_DIR

command -v jq || exit 2