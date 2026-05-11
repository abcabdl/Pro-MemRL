#!/usr/bin/env bash
set -euo pipefail

# ===== 鎸夐渶淇敼浠ヤ笅鍙橀噺 =====
AGENT_TYPE="${AGENT_TYPE:-general_e2e}"            # 浣跨敤閫氱敤绔埌绔?agent
TASK="${TASK:-ALL}"                                # 閰嶅悎 TASK_TAGS锛屼粎璇勬祴鎸囧畾鏍囩浠诲姟
TASK_TAGS="${TASK_TAGS:-routine,preference}"       # 榛樿鍙瘎娴?routine / preference
MODEL_NAME="${MODEL_NAME:-google/gemini-3.1-pro-preview}"   # Gemini 妯″瀷鍚?LLM_BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}" # 蹇呴』鏄?OpenAI-compatible 鍦板潃
GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-REPLACE_WITH_YOUR_API_KEY}}"            # 浼樺厛璇诲彇 GEMINI_API_KEY锛屽叾娆¤鍙?GOOGLE_API_KEY
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"            # 骞跺彂鏁帮紝寤鸿涓嶈秴杩囧彲鐢ㄧ幆澧冩暟
MAX_ROUND="${MAX_ROUND:-50}"                       # 姣忎釜浠诲姟鏈€澶氫氦浜掕疆鏁?STEP_WAIT_TIME="${STEP_WAIT_TIME:-5}"              # 姣忔鍚庣殑绛夊緟鏃堕棿
ENV_IMAGE="${ENV_IMAGE:-ghcr.io/yaosqz/knowu-bench:latest}" # 鑷姩鍙戠幇瀹瑰櫒鏃朵娇鐢ㄧ殑榛樿闀滃儚
AW_HOST="${AW_HOST:-http://127.0.0.1:6800,http://127.0.0.1:6801,http://127.0.0.1:6802,http://127.0.0.1:6803,http://127.0.0.1:6804,http://127.0.0.1:6805,http://127.0.0.1:6806,http://127.0.0.1:6807}" # 澶氱幆澧冨湴鍧€锛涚暀绌哄彲鑷姩鍙戠幇
USER_FILTER="${USER_FILTER:-}"                     # 鍙€? user / student / developer / grandma
USER_LOG_SOURCE="${USER_LOG_SOURCE:-noise}"        # clean / noise
USER_LOG_MODE="${USER_LOG_MODE:-all}"              # all / rag
RAG_TOP_K="${RAG_TOP_K:-10}"
RAG_BACKEND="${RAG_BACKEND:-embedding}"            # tfidf / embedding
ENABLE_MCP="${ENABLE_MCP:-false}"                  # true 鏃堕澶栫撼鍏ュ甫 agent-mcp 鐨?routine/preference 浠诲姟
# ============================

AGENT_API_KEY="$GEMINI_API_KEY"
if [[ -z "$AGENT_API_KEY" || "$AGENT_API_KEY" == "REPLACE_WITH_YOUR_API_KEY" ]]; then
    echo "璇峰厛璁剧疆 GEMINI_API_KEY 鎴?GOOGLE_API_KEY銆? >&2
    exit 1
fi

if [[ "$LLM_BASE_URL" == "[gemini_openai_compatible_base_url]" ]]; then
    echo "璇峰厛璁剧疆 LLM_BASE_URL 涓?Gemini 鐨?OpenAI-compatible 鏈嶅姟鍦板潃銆? >&2
    exit 1
fi

# routine / preference 浠诲姟閫氬父浼氳Е鍙?ask-user 鎴?preference judge銆?# 濡傛灉浣犳病鏈夊崟鐙寚瀹?USER_AGENT_*锛岃繖閲岄粯璁ゅ鐢ㄤ富妯″瀷閰嶇疆锛屼繚璇佽剼鏈彲鐩存帴杩愯銆?export USER_AGENT_API_KEY="${USER_AGENT_API_KEY:-$AGENT_API_KEY}"
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
    --enable-user-interaction \
    --max_round "$MAX_ROUND" \
    --model_name "$MODEL_NAME" \
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
