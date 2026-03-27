#!/usr/bin/env bash
set -xeuo pipefail
GYM_ROOT="${GYM_ROOT:-/mnt/code/yehangcheng/Gym}"
GYM_VERIFY_HOST="${GYM_VERIFY_HOST:-127.0.0.1}"
GYM_VERIFY_PORT="${GYM_VERIFY_PORT:-18001}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-Qwen/Qwen3-4B}"
cd "${GYM_ROOT}"
if [[ -f ".venv/bin/activate" ]]; then source .venv/bin/activate; fi
cat > env.yaml <<EOF
policy_base_url: ${POLICY_BASE_URL}
policy_api_key: ${POLICY_API_KEY}
policy_model_name: ${POLICY_MODEL_NAME}
EOF
config_paths="resources_servers/math_with_judge/configs/dapo17k.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]" \
  +math_with_judge.resources_servers.math_with_judge.should_use_judge=false \
  +math_with_judge.resources_servers.math_with_judge.host="${GYM_VERIFY_HOST}" \
  +math_with_judge.resources_servers.math_with_judge.port="${GYM_VERIFY_PORT}"
