#!/usr/bin/env bash
set -euo pipefail

# ===== 鎸夐渶淇敼浠ヤ笅鍙橀噺 =====
AGENT_TYPE="${AGENT_TYPE:-qwen3.5}"                    # 缁х画浣跨敤 Qwen3.5 agent
TASK="${TASK:-ALL}"                                    # 瑕佽瘎娴嬬殑浠诲姟锛孉LL 琛ㄧず杩愯鍏ㄩ儴浠诲姟
TASK_TAGS="${TASK_TAGS:-routine,preference,general}"           # 鎸夋爣绛剧瓫閫変换鍔★紝閫楀彿鍒嗛殧澶氫釜鏍囩
MODEL_NAME="${MODEL_NAME:-qwen/qwen3.5-122b-a10b}"     # OpenRouter 涓婄殑妯″瀷鍚?LLM_BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}" # OpenAI-compatible 鍦板潃
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-REPLACE_WITH_YOUR_API_KEY}"           # 寤鸿閫氳繃鐜鍙橀噺浼犲叆
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"                # 鏈€澶у苟鍙戣瘎娴嬩换鍔℃暟
MAX_ROUND="${MAX_ROUND:-50}"                           # 姣忎釜浠诲姟鐨勬渶澶т氦浜掕疆鏁?STEP_WAIT_TIME="${STEP_WAIT_TIME:-10}"                 # 姣忔鎿嶄綔鍚庣殑绛夊緟鏃堕棿锛堢锛?ENV_IMAGE="${ENV_IMAGE:-ghcr.io/yaosqz/knowu-bench:latest}" # 鑷姩鍙戠幇瀹瑰櫒鏃朵娇鐢ㄧ殑榛樿闀滃儚
AW_HOST="${AW_HOST:-http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803,http://127.0.0.1:6804,http://127.0.0.1:6805,http://127.0.0.1:6806,http://127.0.0.1:6807}"
USER_FILTER="${USER_FILTER:-}"                         # 鐣欑┖琛ㄧず璺戞墍鏈夌敤鎴凤紱濡傞渶鍗曠敤鎴峰彲璁句负 user/student/developer/grandma
USER_LOG_SOURCE="${USER_LOG_SOURCE:-noise}"            # 鐢ㄦ埛鏃ュ織鏉ユ簮: clean / noise
USER_LOG_MODE="${USER_LOG_MODE:-all}"                  # 鐢ㄦ埛鏃ュ織娉ㄥ叆妯″紡: all / rag
RAG_TOP_K="${RAG_TOP_K:-10}"                           # RAG 妫€绱㈣繑鍥炲墠 K 鏉℃棩蹇?RAG_BACKEND="${RAG_BACKEND:-embedding}"                # RAG 鍚庣: tfidf / embedding
ENABLE_MCP="${ENABLE_MCP:-false}"                      # true 鏃堕檮鍔?--enable_mcp
# ============================

AGENT_API_KEY="REPLACE_WITH_YOUR_API_KEY"
if [[ -z "$AGENT_API_KEY" || "$AGENT_API_KEY" == "REPLACE_WITH_YOUR_API_KEY" ]]; then
    echo "璇峰厛璁剧疆 OPENROUTER_API_KEY銆? >&2
    exit 1
fi

# routine / preference 浠诲姟閫氬父浼氳Е鍙?ask-user 鎴?preference judge銆?# 濡傛灉娌℃湁鍗曠嫭鎸囧畾 USER_AGENT_*锛岃繖閲岄粯璁ゅ鐢ㄤ富妯″瀷閰嶇疆銆?export USER_AGENT_API_KEY="${USER_AGENT_API_KEY:-$AGENT_API_KEY}"
export USER_AGENT_BASE_URL="${USER_AGENT_BASE_URL:-$LLM_BASE_URL}"
export USER_AGENT_MODEL="${USER_AGENT_MODEL:-$MODEL_NAME}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1,10.130.138.46,10.130.138.47,10.130.138.48}"

MODEL_TAG="${MODEL_NAME//\//_}"
MODEL_TAG="${MODEL_TAG//./_}"
MODEL_TAG="${MODEL_TAG//-/_}"
TASK_TAGS_TAG="${TASK_TAGS//,/_}"
USER_TAG="${USER_FILTER:-all_users}"
MCP_TAG="no_mcp"
if [[ "$ENABLE_MCP" == "true" ]]; then
    MCP_TAG="with_mcp"
fi

LOG_ROOT="traj_logs/${MODEL_TAG}_${TASK_TAGS_TAG}_${USER_TAG}_${USER_LOG_SOURCE}_${USER_LOG_MODE}_${RAG_BACKEND}_${MCP_TAG}"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/nohup_$(date +%Y%m%d_%H%M%S).log"

USER_ARGS=()
if [[ -n "$USER_FILTER" ]]; then
    USER_ARGS=(--user "$USER_FILTER")
fi

MCP_ARGS=()
if [[ "$ENABLE_MCP" == "true" ]]; then
    MCP_ARGS=(--enable_mcp)
fi

AW_HOST_ARGS=()
if [[ -n "$AW_HOST" ]]; then
    AW_HOST_ARGS=(--aw-host "$AW_HOST")
fi

nohup mw eval \
    --agent_type "$AGENT_TYPE" \
    --task "$TASK" \
    --task-tags "$TASK_TAGS" \
    --max_round "$MAX_ROUND" \
    --model_name "$MODEL_NAME" \
    --enable-user-interaction \
    --llm_base_url "$LLM_BASE_URL" \
    --api_key "$AGENT_API_KEY" \
    --step_wait_time "$STEP_WAIT_TIME" \
    --env-image "$ENV_IMAGE" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --log_file_root "$LOG_ROOT" \
    "${AW_HOST_ARGS[@]}" \
    --user-log-source "$USER_LOG_SOURCE" \
    --user-log-mode "$USER_LOG_MODE" \
    --rag-top-k "$RAG_TOP_K" \
    --rag-backend "$RAG_BACKEND" \
    "${MCP_ARGS[@]}" \
    "${USER_ARGS[@]}" > "$RUN_LOG" 2>&1 &

echo "浠诲姟宸插湪鍚庡彴鍚姩 PID=$!"
echo "涓绘ā鍨? $MODEL_NAME"
echo "Agent 绫诲瀷: $AGENT_TYPE"
echo "Base URL: $LLM_BASE_URL"
echo "浠诲姟鏍囩: $TASK_TAGS"
echo "骞跺彂鏁? $MAX_CONCURRENCY"
echo "鏃ュ織鏂囦欢: $RUN_LOG"
echo "USER_AGENT_MODEL: $USER_AGENT_MODEL"
if [[ -n "$AW_HOST" ]]; then
    echo "鍚庣鍦板潃: $AW_HOST"
else
    echo "鍚庣鍦板潃: 鑷姩鍙戠幇 knowu_bench_env_* 瀹瑰櫒锛堥暅鍍忚繃婊? $ENV_IMAGE锛?
fi
