for step in 10 20 30; do
    STEP_DIR="${BASE_DIR}/global_step_${step}"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${STEP_DIR}/actor" \
        --target_dir "${STEP_DIR}/result"
done
