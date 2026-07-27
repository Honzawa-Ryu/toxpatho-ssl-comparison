# ==========================================
# Experiment Management Shortcuts
# ==========================================

# Move to the Git root of the current repository.
groot() {
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null)
    if [ -n "$root" ]; then
        cd "$root" || return
        echo "🌳 Moved to Git root: $(basename "$root")"
    else
        echo "❌ Error: Not inside a Git repository."
    fi
}

# Helper function to dynamically find the Git root AT EXECUTION TIME
_get_dynamic_root() {
    local root
    root=$(git rev-parse --show-toplevel 2>/dev/null)
    if [ -z "$root" ]; then
        echo ""
    else
        echo "$root"
    fi
}

# List experiments
lsx() {
    local root=$(_get_dynamic_root)
    if [ -z "$root" ]; then
        echo "❌ Error: Not inside a Git repository."
        return 1
    fi

    echo "🔍 Latest Experiments in [$(basename "$root")]:"

    ls -d "${root}/experiments/"*/ 2>/dev/null \
        | sort -r \
        | head -n 10 \
        | awk -F/ '
            {
                name=$(NF-1)
                split(name, arr, "_")
                id=arr[1]
                printf "%-6s %s\n", id, name
            }
        '
}

# CD into experiment
cdx() {
    local root=$(_get_dynamic_root)
    if [ -z "$root" ]; then
        echo "❌ Error: Not inside a Git repository."
        return 1
    fi

    local target=""

    if [[ "$1" =~ ^[0-9]+$ ]]; then
        local padded
        padded=$(printf "%04d" "$((10#$1))")
        target=$(
            ls -d "${root}/experiments/${padded}_"*/ 2>/dev/null \
            | head -n1
        )
    else
        target=$(
            ls -d "${root}/experiments/"*/ 2>/dev/null \
            | sort -r \
            | head -n1
        )
    fi

    if [ -z "$target" ]; then
        echo "❌ Experiment not found."
        return 1
    fi

    cd "$target" || return
    echo "📂 Moved to: $(basename "$target")"
}

# Submit experiment
#
# run_slurm.sh 自体が完全な #SBATCH ヘッダーを持つ投入スクリプトなので、
# runx はそれをそのまま sbatch するだけ。partition/リソース/array-seq切り替え/
# 依存ジョブ指定（#SBATCH --dependency=）はすべて run_slurm.sh 側に書かれている
# （make create_exp 時に一度だけ partition/signal marginが焼き込まれる。
#  変えたい場合は run_slurm.sh を直接編集する）。
runx() {
    local root
    root=$(git rev-parse --show-toplevel)

    local exp_id=""
    local allow_dirty=0

    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        exp_id="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --allow-dirty)
                allow_dirty=1
                shift
                ;;
            *)
                echo "❌ Unknown option: $1"
                return 1
                ;;
        esac
    done

    if ! git diff --quiet || ! git diff --cached --quiet; then
        if [ "$allow_dirty" -eq 0 ]; then
            echo ""
            echo "❌ Git working tree is dirty."
            echo ""
            echo "Commit changes or use:"
            echo ""
            echo "  runx ... --allow-dirty"
            echo ""
            return 1
        fi
        echo ""
        echo "⚠️ Running with dirty git state"
        echo ""
    fi

    local target=""
    if [ -n "$exp_id" ]; then
        local padded
        padded=$(printf "%04d" "$((10#$exp_id))")
        target=$(
            find "${root}/experiments" \
                -maxdepth 1 \
                -type d \
                -name "${padded}_*" \
            | head -n1
        )
    else
        target=$(
            find "${root}/experiments" \
                -mindepth 1 \
                -maxdepth 1 \
                -type d \
                -printf "%f\n" \
            | grep -E '^[0-9]{4}_' \
            | sort \
            | tail -n1
        )
        target="${root}/experiments/${target}"
    fi

    if [ ! -d "$target" ]; then
        echo "❌ Experiment not found."
        return 1
    fi

    local exp_name
    exp_name=$(basename "$target")

    # #SBATCH --output/--error のディレクトリが無いと sbatch が失敗するため
    mkdir -p "${root}/logs/${exp_name}"

    echo ""
    echo "🚀 Submitting"
    echo ""
    echo "  EXP_NAME : ${exp_name}"
    echo "  SCRIPT   : ${target}/run_slurm.sh"
    echo ""

    local sbatch_output
    sbatch_output=$(sbatch "${target}/run_slurm.sh")
    echo "${sbatch_output}"

    local job_id
    job_id=$(echo "${sbatch_output}" | awk '{print $4}')

    if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "❌ sbatch failed."
        return 1
    fi

    # 他の実験からこのjob_idに依存させたい場合の参照元
    # （run_slurm.shの#SBATCH --dependency=afterok:<job_id>に手動で転記する）
    mkdir -p "${root}/outputs/${exp_name}"
    echo "${job_id}" > "${root}/outputs/${exp_name}/latest_job_id.txt"
}

cancelx() {
    local job_id="${1:-}"
    local reason="${2:-user_cancel}"

    if [ -z "${job_id}" ]; then
        echo "Usage:"
        echo "  cancelx <job_id> [reason]"
        return 1
    fi

    local root
    root=$(git rev-parse --show-toplevel)

    bash \
        "${root}/tools/cancel_job.sh" \
        "${job_id}" \
        "${reason}"
}
