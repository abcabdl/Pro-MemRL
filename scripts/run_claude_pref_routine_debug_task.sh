#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ===== 鎸夐渶淇敼浠ヤ笅鍙橀噺 =====
DEFAULT_DEBUG_TASKS=(
    # Dynamic-time tasks changed in the latest pass.
    "MorningPaperReadingGeneralTask"
    "MorningWeatherCheckGeneralTask"
    "MattermostResponseGeneralTask"
    "NightSocialGeneralTask"
    "SendWeeklyReportGeneralTask"
    "WeeklyReportRoutineTask@user"
    "ClockOutGeneralTask"
    "ClockOutRoutineTask@developer"
    "DailyFamilyCallTask@grandma"
    "DeepWorkRoutineTask@user"
    "MorningPaperReadingTask@user"
    "MorningWeatherCheckTask@grandma"
    "ScamSmsInterceptRoutineTask@user"
    # Routine tasks that exercise Mattermost/Mastodon dynamic time paths.
    "MattermostOnCallTask@developer"
    "NightEyeCareRoutineTask@user"
    # Additional Mattermost smoke coverage.
    "MattermostLeaveNoticeGeneralTask"
    "MattermostLeaveNoticeTask@developer"
    "CommuteLateWithNoticeGeneralTask"
    "LateUrgentRouteWithNoticeTask@developer"
    "CalendarScheduleGroupMeetingGeneralTask"
    "CalendarScheduleGroupMeetingTask@developer"
    # Additional Mastodon smoke coverage.
    "MastodonInterestBoostGeneralTask"
    "MastodonInterestBoostTask@user"
    "MastodonPrivacyDefaultGeneralTask"
    "MastodonPrivacyDefaultTestTask@user"
    "MastodonSharePhotosGeneralTask"
    "MastodonSharePhotosPreferenceAskUserTask@user"
)
DEFAULT_TASK="$(IFS=,; echo "${DEFAULT_DEBUG_TASKS[*]}")"

AGENT_TYPE="${AGENT_TYPE:-general_e2e}"
TASK="${1:-${TASK:-$DEFAULT_TASK}}"                                    # 鏀寔閫楀彿鍒嗛殧鐨勫浠诲姟鍒楄〃
TASK_TAGS="${TASK_TAGS:-}"
MODEL_NAME="${MODEL_NAME:-claude-opus-4-6-20260205}" # 鍚嶇О閲屼繚鐣?claude锛岃Е鍙?Claude 涓撶敤缂╂斁閫昏緫
LLM_BASE_URL="${LLM_BASE_URL:-http://101.37.174.109:8010/v1}" # OpenAI-compatible 涓浆鍦板潃
CLAUDE_API_KEY="${CLAUDE_API_KEY:-REPLACE_WITH_YOUR_API_KEY}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
MAX_ROUND="${MAX_ROUND:-50}"
STEP_WAIT_TIME="${STEP_WAIT_TIME:-4}"
ENV_IMAGE="${ENV_IMAGE:-knowu-bench:latest}"
AW_HOST="${AW_HOST:-http://127.0.0.1:6800,http://127.0.0.1:6802,http://127.0.0.1:6803,http://127.0.0.1:6804}"
USER_FILTER="${USER_FILTER:-}"
USER_LOG_SOURCE="${USER_LOG_SOURCE:-noise}"
USER_LOG_MODE="${USER_LOG_MODE:-all}"
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_BACKEND="${RAG_BACKEND:-embedding}"
ENABLE_MCP="${ENABLE_MCP:-false}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-true}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-traj_logs/debug_dynamic_social}"
BATCH_EVAL="${BATCH_EVAL:-true}"               # true 鏃跺崟涓?mw eval 鍐呯敤 MAX_CONCURRENCY 骞惰鍒嗗彂浠诲姟
PARALLEL="${PARALLEL:-false}"                  # BATCH_EVAL=false 鏃跺彲鐢紱true 琛ㄧず澶氳繘绋嬪苟琛?# ============================

if [[ -z "$TASK" ]]; then
    cat >&2 <<'USAGE'
鐢ㄦ硶:
  bash scripts/run_claude_pref_routine_debug_task.sh <TASK_NAME>[,<TASK_NAME2>,...]

绀轰緥锛堝崟浠诲姟锛?
  bash scripts/run_claude_pref_routine_debug_task.sh 'CommuteRoutingBadWeatherTask@developer'

绀轰緥锛堥粯璁?smoke suite锛? 涓幆澧冨苟琛岋級:
  bash scripts/run_claude_pref_routine_debug_task.sh

绀轰緥锛堝浠诲姟锛岄€楀彿鍒嗛殧锛?
  bash scripts/run_claude_pref_routine_debug_task.sh 'TaskA@user,TaskB@student,TaskC@developer'

绀轰緥锛堢幆澧冨彉閲忔柟寮忥級:
  TASK='TaskA@user,TaskB@student' bash scripts/run_claude_pref_routine_debug_task.sh

绀轰緥锛堝浠诲姟骞惰 + 鍚庡彴杩愯锛?
  TASK='TaskA@user,TaskB@student' PARALLEL=true RUN_IN_BACKGROUND=true bash scripts/run_claude_pref_routine_debug_task.sh

閫夐」:
  BATCH_EVAL=true        鍗曚釜 mw eval 鍐呭苟琛屽垎鍙戜换鍔★紙榛樿锛?  PARALLEL=true           澶氫换鍔″苟琛屾墽琛岋紙榛樿涓茶锛?  RUN_IN_BACKGROUND=true  鍚庡彴杩愯锛屾棩蹇楀啓鍏ユ枃浠?  MAX_CONCURRENCY=N       姣忎釜浠诲姟鍐呴儴鐨勫苟鍙戞暟
USAGE
    exit 1
fi

AGENT_API_KEY="$CLAUDE_API_KEY"
if [[ -z "$AGENT_API_KEY" || "$AGENT_API_KEY" == "REPLACE_WITH_YOUR_API_KEY" ]]; then
    echo "璇峰厛鍦ㄨ剼鏈《閮ㄦ妸 CLAUDE_API_KEY 鏀规垚浣犵殑鐪熷疄 API Key銆? >&2
    exit 1
fi

export USER_AGENT_API_KEY="${USER_AGENT_API_KEY:-$AGENT_API_KEY}"
export USER_AGENT_BASE_URL="${USER_AGENT_BASE_URL:-$LLM_BASE_URL}"
export USER_AGENT_MODEL="${USER_AGENT_MODEL:-$MODEL_NAME}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,10.130.138.46,10.130.138.47,10.130.138.48}"

# ---------- 宸ュ叿鍑芥暟 ----------

make_tag() {
    local s="$1"
    s="${s//\//_}"
    s="${s//@/_at_}"
    s="${s//./_}"
    s="${s//-/_}"
    s="${s// /_}"
    echo "$s"
}

build_cmd() {
    local task="$1"
    local log_root="$2"

    local user_args=()
    if [[ -n "$USER_FILTER" ]]; then
        user_args=(--user "$USER_FILTER")
    fi

    local mcp_args=()
    if [[ "$ENABLE_MCP" == "true" ]]; then
        mcp_args=(--enable_mcp)
    fi

    local aw_host_args=()
    if [[ -n "$AW_HOST" ]]; then
        aw_host_args=(--aw-host "$AW_HOST")
    fi

    local task_tag_args=()
    if [[ -n "$TASK_TAGS" ]]; then
        task_tag_args=(--task-tags "$TASK_TAGS")
    fi

    CMD=(
        mw eval
        --agent_type "$AGENT_TYPE"
        --task "$task"
        --enable-user-interaction
        --max_round "$MAX_ROUND"
        --model_name "$MODEL_NAME"
        --llm_base_url "$LLM_BASE_URL"
        --api_key "$AGENT_API_KEY"
        --step_wait_time "$STEP_WAIT_TIME"
        --env-image "$ENV_IMAGE"
        --max-concurrency "$MAX_CONCURRENCY"
        --log_file_root "$log_root"
        --user-log-source "$USER_LOG_SOURCE"
        --user-log-mode "$USER_LOG_MODE"
        --rag-top-k "$RAG_TOP_K"
        --rag-backend "$RAG_BACKEND"
    )
    CMD+=("${aw_host_args[@]}")
    CMD+=("${task_tag_args[@]}")
    CMD+=("${mcp_args[@]}")
    CMD+=("${user_args[@]}")
}

run_batch_tasks() {
    local model_tag
    model_tag="$(make_tag "$MODEL_NAME")"
    local user_tag="${USER_FILTER:-all_users}"
    local mcp_tag="no_mcp"
    [[ "$ENABLE_MCP" == "true" ]] && mcp_tag="with_mcp"

    local run_id
    run_id="$(date +%Y%m%d_%H%M%S)"
    local log_root="${LOG_ROOT_BASE}/${model_tag}_dynamic_social_smoke_${user_tag}_${USER_LOG_SOURCE}_${USER_LOG_MODE}_${RAG_BACKEND}_${mcp_tag}_${run_id}"
    mkdir -p "$log_root"
    local run_log="$log_root/debug.log"

    build_cmd "$TASK" "$log_root"

    echo "============================================"
    echo "[鎵归噺 smoke] 鍗曚釜 mw eval 鍐呭苟琛岃繍琛?$TASK_COUNT 涓换鍔?
    echo "  骞跺彂鏁? $MAX_CONCURRENCY"
    echo "  鏃ュ織鐩綍: $log_root"
    echo "  鏃ュ織鏂囦欢: $run_log"
    echo "============================================"

    if [[ "$RUN_IN_BACKGROUND" == "true" ]]; then
        nohup "${CMD[@]}" > "$run_log" 2>&1 &
        local pid=$!
        echo "[鎵归噺 smoke] 宸插湪鍚庡彴鍚姩 PID=$pid"
        echo "  tail -f '$run_log'"
        echo "  鍙鍖栫粨鏋? mw logs view --log_dir '$log_root'"
        return 0
    fi

    set +e
    "${CMD[@]}" 2>&1 | tee "$run_log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "[鎵归噺 smoke] 瀹屾垚 (exit=$rc)"
    echo "  鍙鍖栫粨鏋? mw logs view --log_dir '$log_root'"
    return $rc
}

run_single_task() {
    local task="$1"
    local idx="$2"
    local total="$3"

    # --- 鏍￠獙 user 涓€鑷存€?---
    if [[ "$task" == *@* && -n "$USER_FILTER" ]]; then
        local task_user="${task##*@}"
        if [[ "$task_user" != "$USER_FILTER" ]]; then
            echo "[浠诲姟 $idx/$total] 璺宠繃: TASK=$task 鍜?USER_FILTER=$USER_FILTER 涓嶄竴鑷? >&2
            return 1
        fi
    fi

    # --- 鏋勯€犳棩蹇楃洰褰?---
    local model_tag
    model_tag="$(make_tag "$MODEL_NAME")"
    local task_log_tag
    task_log_tag="$(make_tag "$task")"
    local user_tag="${USER_FILTER:-all_users}"
    local mcp_tag="no_mcp"
    [[ "$ENABLE_MCP" == "true" ]] && mcp_tag="with_mcp"

    local run_id
    run_id="$(date +%Y%m%d_%H%M%S)"
    local log_root="${LOG_ROOT_BASE}/${model_tag}_${task_log_tag}_${user_tag}_${USER_LOG_SOURCE}_${USER_LOG_MODE}_${RAG_BACKEND}_${mcp_tag}_${run_id}"
    mkdir -p "$log_root"
    local run_log="$log_root/debug.log"

    build_cmd "$task" "$log_root"

    echo "============================================"
    echo "[浠诲姟 $idx/$total] $task"
    echo "  鏃ュ織鐩綍: $log_root"
    echo "  鏃ュ織鏂囦欢: $run_log"
    echo "============================================"

    if [[ "$RUN_IN_BACKGROUND" == "true" ]]; then
        nohup "${CMD[@]}" > "$run_log" 2>&1 &
        local pid=$!
        echo "[浠诲姟 $idx/$total] 宸插湪鍚庡彴鍚姩 PID=$pid"
        echo "$pid" >> "$PIDS_FILE"
    else
        set +e
        "${CMD[@]}" 2>&1 | tee "$run_log"
        local rc=${PIPESTATUS[0]}
        set -e
        echo "[浠诲姟 $idx/$total] 瀹屾垚 (exit=$rc)"
        echo "  鍙鍖栫粨鏋? mw logs view --log_dir '$log_root'"
        return $rc
    fi
}

# ---------- 瑙ｆ瀽浠诲姟鍒楄〃 ----------

IFS=',' read -ra TASKS <<< "$TASK"
TASK_COUNT=${#TASKS[@]}

echo "========================================"
echo "鎵归噺璇勪及鍚姩"
echo "  浠诲姟鏁伴噺: $TASK_COUNT"
echo "  浠诲姟鍒楄〃: ${TASKS[*]}"
echo "  涓绘ā鍨? $MODEL_NAME"
echo "  Agent 绫诲瀷: $AGENT_TYPE"
echo "  Base URL: $LLM_BASE_URL"
echo "  鏈€澶ц疆鏁? $MAX_ROUND"
echo "  鍗曚换鍔″苟鍙? $MAX_CONCURRENCY"
echo "  鎵归噺骞跺彂妯″紡: $BATCH_EVAL"
echo "  澶氫换鍔″苟琛? $PARALLEL"
echo "  鍚庡彴杩愯: $RUN_IN_BACKGROUND"
echo "  USER_AGENT_MODEL: $USER_AGENT_MODEL"
if [[ -n "$TASK_TAGS" ]]; then
    echo "  浠诲姟鏍囩杩囨护: $TASK_TAGS"
fi
if [[ -n "$AW_HOST" ]]; then
    echo "  鍚庣鍦板潃: $AW_HOST"
else
    echo "  鍚庣鍦板潃: 鑷姩鍙戠幇 knowu_bench_env_* 瀹瑰櫒锛堥暅鍍忚繃婊? $ENV_IMAGE锛?
fi
echo "========================================"

# ---------- 鎵ц ----------

PIDS_FILE="$(mktemp)"
trap 'rm -f "$PIDS_FILE"' EXIT

FAILED=0
IDX=0

if [[ "$BATCH_EVAL" == "true" ]]; then
    run_batch_tasks
    exit $?
elif [[ "$PARALLEL" == "true" ]]; then
    # --- 骞惰妯″紡锛氭墍鏈変换鍔″悓鏃跺惎鍔?---
    # 骞惰妯″紡寮哄埗鍚庡彴鍐欐棩蹇楋紙鍓嶅彴 tee 澶氳繘绋嬩細娣蜂贡锛?    RUN_IN_BACKGROUND=true
    for task in "${TASKS[@]}"; do
        task="$(echo "$task" | xargs)"  # trim 绌烘牸
        [[ -z "$task" ]] && continue
        ((IDX+=1))
        run_single_task "$task" "$IDX" "$TASK_COUNT" || true
    done

    # 绛夊緟鎵€鏈夊悗鍙拌繘绋?    echo ""
    echo "绛夊緟鎵€鏈夊悗鍙颁换鍔″畬鎴?.."
    while IFS= read -r pid; do
        if wait "$pid" 2>/dev/null; then
            echo "  PID $pid 瀹屾垚"
        else
            echo "  PID $pid 澶辫触鎴栧凡閫€鍑?
            ((FAILED+=1))
        fi
    done < "$PIDS_FILE"
else
    # --- 涓茶妯″紡锛氶€愪釜鎵ц ---
    for task in "${TASKS[@]}"; do
        task="$(echo "$task" | xargs)"
        [[ -z "$task" ]] && continue
        ((IDX+=1))
        if ! run_single_task "$task" "$IDX" "$TASK_COUNT"; then
            ((FAILED+=1))
        fi
    done
fi

# ---------- 姹囨€?----------

echo ""
echo "========================================"
echo "鍏ㄩ儴瀹屾垚: $TASK_COUNT 涓换鍔? $FAILED 涓け璐?
if [[ "$RUN_IN_BACKGROUND" == "true" && "$PARALLEL" != "true" ]]; then
    echo "鍚庡彴 PID 鍒楄〃:"
    cat "$PIDS_FILE" 2>/dev/null | while read -r p; do echo "  $p"; done
fi
echo "========================================"

exit "$FAILED"
