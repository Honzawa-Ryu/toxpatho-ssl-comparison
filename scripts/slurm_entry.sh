#!/bin/bash
# このファイルは run_slurm.sh の末尾から `source` される前提で書かれている
# （run_slurm.sh 自体が #SBATCH ヘッダーを持つ実際の投入スクリプトであり、
# RUN_MODE/RUN_COMMAND/GRID_ARGS 等は既にその時点で定義済みのため、ここで
# run_slurm.sh を再度 source する必要はない。再度sourceすると、run_slurm.sh
# 末尾のこのファイルへのsource呼び出し自体も再実行され、循環してしまう）。

set -euo pipefail

# =====================================================
# Required variables
#
# ここより下は set -u なので、未定義のまま進むと「unbound variable」だけを
# 残してジョブが無言で死ぬ。ERR trap を張るのは後（Signal handlers 節）なので、
# メタデータもSlack通知も残らない。原因の分かるメッセージで先に落とす。
# =====================================================

: "${PROJECT_ROOT:?PROJECT_ROOT が未設定です。run_slurm.sh から source してください}"
: "${EXP_NAME:?EXP_NAME が未設定です。run_slurm.sh から source してください}"
: "${SIF_PATH:?SIF_PATH（apptainerのSIFイメージのパス）が未設定です。~/.bashrc で export してください（README.md「セットアップ」参照）}"

# =====================================================
# Config
# =====================================================

source "${PROJECT_ROOT}/scripts/notify_slack.sh"

# =====================================================
# Scheduler abstraction
# =====================================================
if [ -n "${PBS_JOBID:-}" ]; then
    export SCHEDULER="pbs"
    export JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1)
    
    if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
        export ARRAY_JOB_ID=$(echo "${PBS_JOBID}" | cut -d. -f1 | cut -d'[' -f1)
        export ARRAY_TASK_ID="${PBS_ARRAY_INDEX}"
    else
        export ARRAY_JOB_ID=""
        export ARRAY_TASK_ID=""
    fi
    export JOB_PARTITION="${PBS_QUEUE:-unknown}"
    export NODENAME=$(hostname)
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    export SCHEDULER="slurm"
    export JOB_ID="${SLURM_JOB_ID}"
    export ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-}"
    export ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
    export JOB_PARTITION="${SLURM_JOB_PARTITION:-unknown}"
    export NODENAME="${SLURMD_NODENAME:-unknown}"
else
    export SCHEDULER="local"
    export JOB_ID="$$"
    export ARRAY_JOB_ID=""
    export ARRAY_TASK_ID=""
    export JOB_PARTITION="local"
    export NODENAME=$(hostname)
fi

# =====================================================
# Run mode
#   single : RUN_COMMAND を1つ実行（デフォルト）
#   array  : ARRAY_TASK_ID で CONFIGS を引く
#   seq    : CONFIGS をループして順番に実行
# =====================================================

RUN_MODE="${RUN_MODE:-single}"

# run_slurm.sh の時間制限を直接読む（run_slurm.sh はヘッダーにしか時間制限を持たず、
# DEFAULT_TIME のようなbash変数はもう存在しないため）。
#
# PBS スクリプトは `#SBATCH --time=` ではなく `#PBS -l walltime=` を持つので
# 両方を試す。また、これは通知の表示用の値でしかないので、どちらにもヒットしない
# 場合でもジョブを落としてはいけない（grep の exit 1 が pipefail + set -e で
# 致命傷になり、ERR trap 設置前なので無言死する）。
_read_job_time_limit() {
    local run_script="${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh"
    local value=""

    [ -f "${run_script}" ] || { echo "unknown"; return 0; }

    value=$(grep -oP '(?<=^#SBATCH --time=)\S+' "${run_script}" | head -n1 || true)
    if [ -z "${value}" ]; then
        value=$(grep -oP '(?<=^#PBS -l walltime=)\S+' "${run_script}" | head -n1 || true)
    fi

    echo "${value:-unknown}"
}

JOB_TIME_LIMIT=$(_read_job_time_limit)
export JOB_TIME_LIMIT

# =====================================================
# Directories
# =====================================================

# array の場合はジョブIDにarray_idを付与
if [ "${RUN_MODE}" = "array" ]; then
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
else
    export JOB_LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}/${JOB_ID}"
fi

mkdir -p "${JOB_LOG_DIR}"

ln -sfn "${JOB_LOG_DIR}" \
    "${PROJECT_ROOT}/logs/${EXP_NAME}/latest"

export SLURM_LOG_FILE="${JOB_LOG_DIR}/slurm.out"

# =====================================================
# Metadata helper
#
# run_metadata.yaml はフラットな key: value しか持たないので、読み書きに
# Python も PyYAML も使わない。ここは apptainer の外・.venv の外で走る
# ネイティブ bash レイヤーであり、ログインノード/計算ノードのシステム python に
# yaml が入っている保証が無いため、依存すると全ジョブがメタデータ初期化で落ちる。
#
# 値は必ずダブルクォートで括る。`196:00:00` のような値は、クォートしないと
# YAML 1.1 の 60進数として整数に解釈されてしまう。
# =====================================================

_yaml_escape() {
    # ダブルクォート内で意味を持つ文字だけを潰す
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

_yaml_put() {
    local yaml_file="$1"
    local key="$2"
    local value
    value=$(_yaml_escape "$3")

    if [ -f "${yaml_file}" ] && grep -q "^${key}:" "${yaml_file}"; then
        # 既存キーは行の位置を保ったまま置換する
        local tmp="${yaml_file}.tmp.$$"
        awk -v k="${key}" -v v="${value}" '
            $0 ~ "^" k ":" { printf "%s: \"%s\"\n", k, v; next }
            { print }
        ' "${yaml_file}" > "${tmp}"
        mv "${tmp}" "${yaml_file}"
    else
        printf '%s: "%s"\n' "${key}" "${value}" >> "${yaml_file}"
    fi
}

update_metadata() {
    _yaml_put "${JOB_LOG_DIR}/run_metadata.yaml" "$1" "$2"
}

# =====================================================
# Metadata
# =====================================================

GIT_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo "git_not_available")

: > "${JOB_LOG_DIR}/run_metadata.yaml"

update_metadata "exp_name"   "${EXP_NAME}"
update_metadata "job_id"     "${JOB_ID}"
update_metadata "partition"  "${JOB_PARTITION}"
update_metadata "node"       "${NODENAME}"
update_metadata "git_commit" "${GIT_COMMIT}"
update_metadata "start_time" "$(date --iso-8601=seconds)"
update_metadata "time_limit" "${JOB_TIME_LIMIT}"
update_metadata "run_mode"   "${RUN_MODE}"
update_metadata "status"     "RUNNING"

if [ "${RUN_MODE}" = "array" ]; then
    update_metadata "array_task_id" "${ARRAY_TASK_ID}"
fi

# =====================================================
# Move Slurm log helper
# =====================================================

_move_slurm_log() {
    local log_file=""

    if [ "${SCHEDULER}" = "pbs" ]; then
        local pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.o${JOB_ID}"
        fi
        
        log_file=$(ls ${pattern}* 2>/dev/null | head -n1 || true)
        
        if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
        
        local err_pattern=""
        if [ "${RUN_MODE}" = "array" ]; then
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${ARRAY_JOB_ID}-${ARRAY_TASK_ID}"
        else
            err_pattern="${PROJECT_ROOT}/logs/${EXP_NAME}/${EXP_NAME}.e${JOB_ID}"
        fi
        local err_log_file
        err_log_file=$(ls ${err_pattern}* 2>/dev/null | head -n1 || true)
        if [ -n "${err_log_file}" ] && [ -f "${err_log_file}" ]; then
            mv "${err_log_file}" "${JOB_LOG_DIR}/pbs.err"
        fi
    else
        if [ "${RUN_MODE}" = "array" ]; then
            # runx が --output=%A_%a_${exp_name}.out で投入するため、
            # %A（親のarray job id = ARRAY_JOB_ID）を使う。
            # JOB_ID は各タスク固有の値で %A とは異なるため使わない。
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${ARRAY_JOB_ID}_${ARRAY_TASK_ID}_${EXP_NAME}.out"
        else
            log_file="${PROJECT_ROOT}/logs/${EXP_NAME}/${JOB_ID}_${EXP_NAME}.out"
        fi

        if [ -f "${log_file}" ]; then
            mv "${log_file}" "${JOB_LOG_DIR}/slurm.out"
        fi
    fi
}

# =====================================================
# sacct helper
# =====================================================

_get_job_state() {
    if [ "${SCHEDULER}" = "pbs" ]; then
        if command -v qstat &>/dev/null; then
            local target_id="${JOB_ID}"
            if [ "${RUN_MODE}" = "array" ]; then
                target_id="${ARRAY_JOB_ID}[${ARRAY_TASK_ID}]"
            fi
            
            local qstat_out
            qstat_out=$(qstat -f "${target_id}" 2>/dev/null || true)
            
            if [ -z "${qstat_out}" ]; then
                echo "COMPLETED"
                return
            fi
            
            local comment
            comment=$(echo "${qstat_out}" | grep -E "comment =" || true)
            
            if echo "${comment}" | grep -iq "walltime"; then
                echo "TIMEOUT"
                return
            fi
            if echo "${comment}" | grep -iq -E "oom|out of memory"; then
                echo "OUT_OF_MEMORY"
                return
            fi
            
            local exit_status
            exit_status=$(echo "${qstat_out}" | grep -E "Exit_status =" | awk '{print $3}' || echo "")
            if [ "${exit_status}" = "271" ]; then
                echo "TIMEOUT"
            elif [ "${exit_status}" = "137" ]; then
                echo "OUT_OF_MEMORY"
            elif [ "${exit_status}" = "0" ] || [ -z "${exit_status}" ]; then
                echo "COMPLETED"
            else
                echo "FAILED"
            fi
        else
            echo "UNKNOWN"
        fi
    elif [ "${SCHEDULER}" = "local" ]; then
        # SCHEDULER=local には問い合わせ先が無い。終了コードだけが判断材料なので
        # COMPLETED を返し、_handle_final_state 側の exit_code 判定に委ねる。
        echo "COMPLETED"
    else
        local state=""
        if command -v sacct &>/dev/null; then
            state=$(
                sacct \
                    -j "${JOB_ID}" \
                    --format=JobIDRaw,State \
                    --parsable2 \
                    --noheader \
                | awk -F'|' -v id="${JOB_ID}" '$1==id {print $2; exit}'
            )
        elif command -v scontrol &>/dev/null; then
            state=$(
                scontrol show job "${JOB_ID}" 2>/dev/null \
                    | grep -oP 'JobState=\K\w+' || true
            )
        fi
        echo "${state:-UNKNOWN}"
    fi
}

_handle_final_state() {
    trap - ERR
    local exit_code="${1:-0}"
    local line_no="${2:-unknown}"

    sleep 3

    local job_state
    job_state=$(_get_job_state)

    case "${job_state}" in

        TIMEOUT)
            update_metadata "status" "TIMEOUT"
            update_metadata "fail_reason" "SLURM_TIMEOUT"
            notify_fail "TIMEOUT"
            ;;

        OUT_OF_MEMORY)
            update_metadata "status" "OUT_OF_MEMORY"
            update_metadata "fail_reason" "OUT_OF_MEMORY"
            notify_fail "OUT_OF_MEMORY"
            ;;

        CANCELLED*)
            update_metadata "status" "CANCELLED"
            update_metadata "fail_reason" "CANCELLED"
            notify_fail_fast "CANCELLED"
            ;;

        NODE_FAIL)
            update_metadata "status" "NODE_FAIL"
            update_metadata "fail_reason" "NODE_FAIL"
            notify_fail "NODE_FAIL"
            ;;

        COMPLETED)
            if [ "${exit_code}" -eq 0 ]; then
                update_metadata "status" "COMPLETED"
                notify_finish
            else
                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"
                notify_fail "NONZERO_EXIT (code ${exit_code})"
            fi
            ;;

        # UNKNOWN = sacct/scontrol/qstat に問い合わせられなかった（またはまだ
        # 状態が確定していない）ケース。状態が取れないことそれ自体は失敗ではないので、
        # RUNNING 等と同じく本体コマンドの終了コードで判定する。
        RUNNING|COMPLETING|CONFIGURING|UNKNOWN)
            if [ "${exit_code}" -eq 0 ]; then

                update_metadata "status" "COMPLETED"
                notify_finish

            else

                update_metadata "status" "FAILED"
                update_metadata "fail_reason" "NONZERO_EXIT_${exit_code}"

                notify_fail "NONZERO_EXIT (code ${exit_code})"

            fi
            ;;

        *)
            update_metadata "status" "FAILED"
            update_metadata "fail_reason" "SCRIPT_ERROR_LINE_${line_no}_CODE_${exit_code}"
            notify_fail "SCRIPT_ERROR (line ${line_no}, code ${exit_code}, state ${job_state})"
            ;;

    esac

    _move_slurm_log
}

# =====================================================
# Signal handlers
# =====================================================

on_error() {
    trap - ERR
    local exit_code=$?
    local line_no="${1:-unknown}"

    echo ""
    echo "❌ Error on line ${line_no} (exit code: ${exit_code})"
    echo ""

    _handle_final_state "${exit_code}" "${line_no}"
}

trap 'on_error ${LINENO}' ERR

TERMINATED=0

on_terminate() {
    trap - TERM INT
    TERMINATED=1

    echo "TERM/INT RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"

    _handle_final_state 1 "signal"
}

trap on_terminate TERM INT

# =====================================================
# Time limit warning (旧 watch_job.sh の代替)
# --signal=B:USR1@<margin> により、Slurm が残り時間僅少になった時点で
# このバッチスクリプト自身に USR1 を送る。別プロセスのwatcherは使わない。
# ジョブ自体は継続する（通知のみ、_handle_final_state は呼ばない）。
# =====================================================

on_time_limit_warning() {
    echo "TIME_LIMIT_WARNING RECEIVED $(date)" \
        >> "${JOB_LOG_DIR}/signal_debug.log"
    # ⚡ INTERRUPTED (notify_fail_fast) は使わない。README の通知表では
    # ⚡ =「キャンセル・割り込み」であり、ジョブが継続しているのに中断したと
    # 誤解される。⏳ 専用の通知を送る。
    notify_time_limit_warning "TIME_LIMIT_WARNING"
}

trap on_time_limit_warning USR1

# =====================================================
# GRID → CONFIGS 展開
# =====================================================

if [ "${RUN_MODE}" != "single" ] \
   && [[ -v GRID_ARGS && "${#GRID_ARGS[@]}" -gt 0 ]]; then

    # GRID_ARGS / GRID_VALUES から直積を生成して CONFIGS に展開
    _expand_grid() {
        local -n _args=$1
        local -n _values=$2
        local -a result=("")

        for i in "${!_args[@]}"; do
            local arg="${_args[$i]}"
            local -a vals
            read -ra vals <<< "${_values[$i]}"

            local -a next=()

            for prev in "${result[@]}"; do
                for val in "${vals[@]}"; do
                    if [ -z "${prev}" ]; then
                        next+=("${arg} ${val}")
                    else
                        next+=("${prev} ${arg} ${val}")
                    fi
                done
            done

            result=("${next[@]}")
        done

        printf '%s\n' "${result[@]}"
    }

    mapfile -t CONFIGS < <(_expand_grid GRID_ARGS GRID_VALUES)
fi

# =====================================================
# Resolve run command
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then

    # CONFIGS[ARRAY_TASK_ID] を引く
    RESOLVED_COMMAND="${BASE_COMMAND} ${CONFIGS[${ARRAY_TASK_ID}]}"

elif [ "${RUN_MODE}" = "seq" ]; then

    # seqの場合はループで実行するのでここではダミー
    RESOLVED_COMMAND=""

else

    RESOLVED_COMMAND="${RUN_COMMAND}"

fi

# =====================================================
# Array info for notifications
# =====================================================

if [ "${RUN_MODE}" = "array" ]; then
    ARRAY_TOTAL_TASKS="${#CONFIGS[@]}"
    ARRAY_TASK_OPTIONS="${CONFIGS[${ARRAY_TASK_ID}]}"
fi

# =====================================================
# Start notify
# =====================================================

notify_start

# =====================================================
# Scratch
#
# SCRATCH_ROOT はノード付属SSDのマウントポイント。既定は /scratch だが、
# 別の場所にマウントしているサイトや、スモークテスト（tests/smoke_slurm_entry.sh）
# のように /scratch が存在しない環境のために上書きできるようにしてある。
# =====================================================

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch}"

export SCRATCH_DIR="${SCRATCH_ROOT}/${USER}/${EXP_NAME}_${JOB_ID}"

mkdir -p "${SCRATCH_DIR}"

# =====================================================
# Input
#
# DATA_SUBDIRS が1つ以上設定されていれば、data/ 全体ではなく列挙された
# サブディレクトリだけを転送する（無駄な転送を避けるため、run_slurm.sh 側で
# 実験に必要な最小限のパスを指定することを推奨）。DATA_SUBDIRS が空の場合は
# 従来通り data/ 全体をコピーする（後方互換）。
# =====================================================

if [ "${USE_LOCAL_SSD_INPUT:-0}" -eq 1 ]; then

    mkdir -p "${SCRATCH_DIR}/data"

    if [[ -v DATA_SUBDIRS && "${#DATA_SUBDIRS[@]}" -gt 0 ]]; then

        for sub in "${DATA_SUBDIRS[@]}"; do
            mkdir -p "${SCRATCH_DIR}/data/$(dirname "${sub}")"
            rsync -a \
                "${PROJECT_ROOT}/data/${sub}" \
                "${SCRATCH_DIR}/data/$(dirname "${sub}")/"
        done

    else

        rsync -a \
            "${PROJECT_ROOT}/data/" \
            "${SCRATCH_DIR}/data/"

    fi

    export DATASET_DIR="${SCRATCH_DIR}/data"

else

    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

# =====================================================
# Output
# =====================================================

export OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"

mkdir -p "${OUTPUT_DIR}"

if [ "${USE_LOCAL_SSD_OUTPUT:-0}" -eq 1 ]; then

    mkdir -p "${SCRATCH_DIR}/outputs"

    # lib/output_utils.get_run_dir() writes here when set; the
    # completed-guard still always checks OUTPUT_DIR (project_root),
    # never this scratch path.
    export OUTPUT_ROOT="${SCRATCH_DIR}/outputs"
fi

# =====================================================
# Save command
# =====================================================

if [ "${RUN_MODE}" = "seq" ]; then
    printf '%s\n' "${CONFIGS[@]/#/${BASE_COMMAND} }" \
        > "${JOB_LOG_DIR}/command.sh"
else
    echo "${RESOLVED_COMMAND}" \
        > "${JOB_LOG_DIR}/command.sh"
fi

# =====================================================
# Run
# =====================================================

_run_single() {
    local cmd="$1"

    set +e

    apptainer exec \
        --nv \
        --bind "${SCRATCH_ROOT}/${USER}" \
        --bind "${SCRATCH_DIR}" \
        --env UV_CACHE_DIR="${SCRATCH_DIR}/.uv_cache" \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail

            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda

            cd ${PROJECT_ROOT}

            ${cmd}
        " 1>&2 &

    local pid=$!
    local code=0

    wait ${pid} || code=$?

    set -e

    echo "${code}"
}

# =====================================================
# Multi-node DDP (PBS)
#
# opt-in: run_slurm.sh 側で `NNODES`(2以上の整数)を設定した場合のみ使う。
# Miyabi-Gは1ノード=GPU1台(GH200)なので、単一ノード内マルチGPU
# (torchrun --standalone)は使えず、複数ノードにまたがるDDPが必須になる
# (詳細根拠は experiments/0020_.../run_slurm.sh 冒頭コメント参照)。
#
# ⚠️ 実機未検証(2026-08-08時点)。pbsdshの正確な引数・NCCL/InfiniBand周りの
# 追加環境変数の要否・PBS_NODEFILEのフォーマットは、このサイトで一度も
# 実行確認していない。experiments/0020_.../ の2ノードスモークテストで
# 先に疎通確認してから本番実験に使うこと。
#
# 方式: cmd（"python foo.py --args" 形式のRUN_COMMAND）から先頭の "python "
# を剥がし、torchrunのrendezvous引数を前置したうえで、pbsdshで全ノードに
# 同一コマンドを配る（c10d rendezvousはjoin順でrank自動割当されるため、
# 全ノードに同一コマンドを配るだけでよく、ノードごとに --node-rank を
# 出し分ける必要はない）。
# =====================================================

_run_multi_node() {
    local cmd="$1"
    local payload="${cmd#python }"

    if [ ! -f "${PBS_NODEFILE:-}" ]; then
        echo "❌ PBS_NODEFILE が見つかりません（PBS以外のスケジューラでは NNODES>1 は未対応）" >&2
        echo "1"
        return
    fi

    mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
    local n_nodes="${#NODES[@]}"
    if [ "${n_nodes}" -ne "${NNODES}" ]; then
        echo "⚠️  PBS_NODEFILEのユニークノード数(${n_nodes})が NNODES(${NNODES})と一致しません。select句とNNODESを合わせること" >&2
    fi

    local master_addr="${NODES[0]}"
    local master_port="${MASTER_PORT:-29500}"

    echo "=== multi-node DDP: ${n_nodes} nodes, master=${master_addr}:${master_port} ===" >&2
    printf '  node: %s\n' "${NODES[@]}" >&2

    # `apptainer` はPATH解決ではなく絶対パスで渡す。pbsdshで起動される
    # リモートタスクのシェルは `module load` を引き継がない(=PATHにapptainerが
    # 無い)ことを2026-08-08、experiments/0020スモークテストの2回目投入
    # (2507294)で実機確認したため。/work配下はNFS共有なので絶対パスで
    # 全ノードから解決できる想定。
    local apptainer_bin
    apptainer_bin=$(command -v apptainer)

    # 入れ子の文字列展開(bash -c "..." の中にさらに bash -c "..." や
    # export FOO="..." のようなダブルクォートを埋め込む方式)は、内側の
    # クォートが外側のクォートと衝突してコマンドが途中で切れる
    # (2026-08-08、experiments/0020スモークテストの3回目投入(2507323)で
    # 実機確認: `export JOB_LOG_DIR="${JOB_LOG_DIR}"` の埋め込みダブル
    # クォートが外側の `bash -c "${node_cmd}"` を早期に閉じてしまい、
    # torchrunが一度も呼ばれないまま空のexportだけで exit 0 していた)。
    # そのため、各階層は「文字列を直接埋め込む」のではなく「共有NFS
    # (/work配下、全ノードから見える)上のスクリプトファイルをパスで渡す」
    # 方式に統一し、ネストした引用符そのものを発生させない。
    # USE_LOCAL_SSD_INPUT=1の場合、メインスクリプトの「Input」節(冒頭付近)は
    # 実行ノード(mother superior)のローカルSSDにしかrsyncしないため、残りの
    # ノードにはデータが存在しない。ここで同じロジックをノードごとに
    # 再実行できるようlauncher_script側に埋め込む。
    local dataset_dir_value="${PROJECT_ROOT}/data"
    local data_staging_cmds="true"
    if [ "${USE_LOCAL_SSD_INPUT:-0}" -eq 1 ]; then
        dataset_dir_value="${SCRATCH_DIR}/data"
        data_staging_cmds="mkdir -p \"${SCRATCH_DIR}/data\""
        if [[ -v DATA_SUBDIRS && "${#DATA_SUBDIRS[@]}" -gt 0 ]]; then
            for sub in "${DATA_SUBDIRS[@]}"; do
                data_staging_cmds="${data_staging_cmds}
mkdir -p \"${SCRATCH_DIR}/data/$(dirname "${sub}")\"
rsync -a \"${PROJECT_ROOT}/data/${sub}\" \"${SCRATCH_DIR}/data/$(dirname "${sub}")/\""
            done
        else
            data_staging_cmds="${data_staging_cmds}
rsync -a \"${PROJECT_ROOT}/data/\" \"${SCRATCH_DIR}/data/\""
        fi
    fi

    local incontainer_script="${JOB_LOG_DIR}/_multinode_incontainer.sh"
    cat > "${incontainer_script}" <<EOF
#!/bin/bash
set -euo pipefail
source ${PROJECT_ROOT}/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export JOB_LOG_DIR=${JOB_LOG_DIR}
export DATASET_DIR=${dataset_dir_value}
# 以下の環境変数はメインスクリプト(run_slurm.sh)がexportしていても、pbsdshの
# リモートタスクへ継承される保証がない(apptainer絶対パス化と同じ理由)。
# PYTHONPATH未継承ではlib以下のimportがModuleNotFoundErrorで落ち、
# WANDB_MODE未継承ではwandb.init()がオフライン扱いにならずAPIキー無しで
# UsageErrorになり rank0 がクラッシュ、それにより他rankもTCPStore接続断で
# 巻き添えクラッシュすることを2026-08-09/10、DINO 4ノードジョブ(2508476,
# 2513143)で実機確認した(この行のコメントにバッククォートを使ったところ、
# 非quotedヒアドキュメント内でコマンド置換として実行されてしまいノイズに
# なったため、この注意書き自体もバッククォート無しにしてある)。
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd ${PROJECT_ROOT}
# 素の torchrun コマンドは /usr/local/bin/torchrun (shebang: /usr/bin/python3、
# コンテナのシステムpython) を指しており、.venv/bin/torchrun は存在しない。
# そのままだとワーカーがシステムpythonで起動し、.venvにだけpip installした
# パッケージ(scikit-learn等)がModuleNotFoundErrorになることを2026-08-09、
# DINO 4ノードジョブ(2509305)で実機確認した。python -m torch.distributed.run
# で起動すればsys.executable経由で.venv/bin/pythonが使われる。
# c10d rendezvousのTCPStore read_timeoutはデフォルト60秒(torch.distributed.
# elastic.rendezvous.c10d_rendezvous_backend.create_backendのdocstringで確認)。
# ノードごとにUSE_LOCAL_SSD_INPUT=1の141GB rsync + apptainerのSIFサンドボックス
# 展開 + 重いimportの所要時間がばらつき、60秒枠を超えて
# RendezvousConnectionError(timed out after 60000ms)になることを2026-08-09、
# DINO 4ノードジョブ(2509371)で実機確認したため、大きめに延長しておく。
python -m torch.distributed.run --nnodes=${n_nodes} --nproc_per_node=${NPROC_PER_NODE:-1} \
    --rdzv_backend=c10d --rdzv_id=${JOB_ID} \
    --rdzv_endpoint=${master_addr}:${master_port} \
    --rdzv_conf=read_timeout=600 \
    ${payload}
EOF
    chmod +x "${incontainer_script}"

    # SCRATCH_DIR(ノードローカルNVMe SSD配下)はメインスクリプトの
    # mkdir -p では実行ノード(mother superior)にしか作られないため、
    # pbsdshで各ノードに配るこの外側スクリプトでノードごとに作り直す。
    # データのrsync(USE_LOCAL_SSD_INPUT=1時)も同様に全ノードで個別に必要
    # (⚠️ 2026-08-08時点、この経路は2ノードスモークテストでは検証していない
    # ―― スモークテストはUSE_LOCAL_SSD_INPUT=0でデータを一切使わないため。
    # 実際の学習ジョブで初めて確認することになる)。
    local launcher_script="${JOB_LOG_DIR}/_multinode_launch.sh"
    cat > "${launcher_script}" <<EOF
#!/bin/bash
set -euo pipefail
echo "[multi-node] \$(hostname) launcher starting"
mkdir -p "${SCRATCH_DIR}"
${data_staging_cmds}
"${apptainer_bin}" exec \
    --nv \
    --bind "${SCRATCH_ROOT}/${USER}" \
    --bind "${SCRATCH_DIR}" \
    --env UV_CACHE_DIR="${SCRATCH_DIR}/.uv_cache" \
    "${SIF_PATH}" \
    bash "${incontainer_script}"
echo "[multi-node] \$(hostname) launcher finished"
EOF
    chmod +x "${launcher_script}"

    set +e

    # pbsdshは `--` 区切りが必須(無いと後続の引数をpbsdsh自身のオプションと
    # して誤パースする。2026-08-08、experiments/0020スモークテストの1回目
    # 投入(2507246)で実機確認)。
    #
    # ⚠️ pbsdsh自体の終了コードは、spawnした子タスクが失敗しても0のままに
    # なる(2026-08-09、DINO 4ノードジョブ(2508476)で実機確認: 4タスク全部が
    # `exit status 1` でも pbsdsh 自体は正常終了し、run_metadata.yamlの
    # statusが実際は失敗しているのに COMPLETED と誤記録された)。そのため
    # `wait`の終了コードを鵜呑みにせず、出力中の `exit status <非ゼロ>` を
    # 自前で検出してcodeへ反映する。
    local pbsdsh_log="${JOB_LOG_DIR}/_pbsdsh_output.log"
    pbsdsh -v -- bash "${launcher_script}" 2>&1 | tee "${pbsdsh_log}" 1>&2 &

    local pid=$!
    local code=0

    wait ${pid} || code=$?

    if grep -qE "exit status [1-9]" "${pbsdsh_log}"; then
        echo "❌ multi-node: 1つ以上のノードでタスクが失敗しました(pbsdshのexit statusを検出)" >&2
        code=1
    fi

    set -e

    echo "${code}"
}

# =====================================================
# Pre-native command (apptainer外で実行したい処理)
# ES 起動など、ネイティブ bash レイヤーで動かすコマンドを
# run_slurm.sh で PRE_NATIVE_COMMAND に設定する
# =====================================================

if [ -n "${PRE_NATIVE_COMMAND:-}" ]; then
    echo "=== PRE_NATIVE_COMMAND ==="
    eval "${PRE_NATIVE_COMMAND}"
fi

EXIT_CODE=0

if [ "${RUN_MODE}" = "seq" ]; then

    # CONFIGS をループして順番に実行
    for i in "${!CONFIGS[@]}"; do

        local_cmd="${BASE_COMMAND} ${CONFIGS[$i]}"

        echo ""
        echo "▶ [${i}/$((${#CONFIGS[@]} - 1))] ${local_cmd}"
        echo ""

        code=$(_run_single "${local_cmd}")

        if [ "${code}" -ne 0 ]; then
            echo "❌ Config [${i}] failed with exit code: ${code}"
            EXIT_CODE="${code}"
            break
        fi
    done

elif [ "${NNODES:-1}" -gt 1 ]; then

    EXIT_CODE=$(_run_multi_node "${RESOLVED_COMMAND}")

else

    EXIT_CODE=$(_run_single "${RESOLVED_COMMAND}")

fi

if [ "${EXIT_CODE}" -ne 0 ]; then
    echo "Child failed with exit code: ${EXIT_CODE}"
fi

# =====================================================
# Post-native command (apptainer外で実行したい後処理)
# ES データの SSD→HDD 同期など
# =====================================================

if [ -n "${POST_NATIVE_COMMAND:-}" ]; then
    echo "=== POST_NATIVE_COMMAND ==="
    eval "${POST_NATIVE_COMMAND}"
fi

# =====================================================
# Output sync-back（scratch → /workspace/outputs）
# USE_LOCAL_SSD_OUTPUT=1 の場合、experiment.py は completion.json を含む
# すべての出力を SCRATCH_DIR/outputs/ 配下に書いている。次回以降の
# completed ガードが正しく機能するよう、scratch を消す前に必ず
# OUTPUT_DIR（project_root 側）へ回収する。
# =====================================================

if [ "${USE_LOCAL_SSD_OUTPUT:-0}" -eq 1 ] \
   && [ -d "${SCRATCH_DIR}/outputs/${EXP_NAME}" ]; then

    rsync -a \
        "${SCRATCH_DIR}/outputs/${EXP_NAME}/" \
        "${OUTPUT_DIR}/"
fi

# =====================================================
# Scratch cleanup
# =====================================================

if [ -n "${SCRATCH_DIR:-}" ] \
   && [ -d "${SCRATCH_DIR}" ] \
   && [[ "${SCRATCH_DIR}" == "${SCRATCH_ROOT}"/* ]]; then

    echo "⚠️  Scratch directory used during this run was NOT auto-deleted: ${SCRATCH_DIR}" >&2
    echo "⚠️  Please remove it manually once no longer needed (node-local SSD capacity)." >&2
fi

# =====================================================
# Skip final status if already handled by signal
# =====================================================

if [ "${TERMINATED}" -eq 1 ]; then
    exit 1
fi

# =====================================================
# Final status
# =====================================================

_handle_final_state "${EXIT_CODE}" "end"
