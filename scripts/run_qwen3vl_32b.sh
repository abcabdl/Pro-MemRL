export DASHSCOPE_API_KEY="REPLACE_WITH_YOUR_API_KEY"  # 寤鸿鍏堣缃幆澧冨彉閲?
LOG_ROOT="traj_logs/qwen3_vl_32b_logs"
mkdir -p "$LOG_ROOT"
RUN_LOG="$LOG_ROOT/nohup_$(date +%Y%m%d_%H%M%S).log"

nohup uv run mw eval \
    --agent_type qwen3vl \
    --task MattermostOnCallTask@user,MorningPaperReadingTask@user,PreMeetingPrepTask@user,WeekendSleeperTask@user,BatterySaverRoutineTask@user,DeepWorkRoutineTask@user \
    --task-tags routine \
    --max_round 50 \
    --model_name qwen3-vl-32b \
    --enable-user-interaction \
    --llm_base_url http://10.130.138.46:8050/v1 \
    --api_key $DASHSCOPE_API_KEY \
    --step_wait_time 10 \
    --max-concurrency 2 \
    --log_file_root "$LOG_ROOT" \
    --aw-host "http://127.0.0.1:6800" > "$RUN_LOG" 2>&1 &

echo "浠诲姟宸插湪鍚庡彴鍚姩 PID=$!"
echo "鏃ュ織鏂囦欢: $RUN_LOG"