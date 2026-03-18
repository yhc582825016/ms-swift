#!/usr/bin/env bash
set -xeuo pipefail
GYM_ROOT="${GYM_ROOT:-/mnt/code/yehangcheng/Gym}"
GYM_VERIFY_HOST="${GYM_VERIFY_HOST:-127.0.0.1}"
GYM_VERIFY_PORT="${GYM_VERIFY_PORT:-18004}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:8000/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"
cd "${GYM_ROOT}"
if [[ -f ".venv/bin/activate" ]]; then source .venv/bin/activate; fi
cat > env.yaml <<EOF
policy_base_url: ${POLICY_BASE_URL}
policy_api_key: ${POLICY_API_KEY}
policy_model_name: ${POLICY_MODEL_NAME}
EOF
config_paths="resources_servers/single_step_tool_use_with_argument_comparison/configs/swe_pivot_single_step_tool_use_with_argument_comparison.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]" \
  +swe_pivot_single_step_tool_use_with_argument_comparison_resources_server.resources_servers.single_step_tool_use_with_argument_comparison.host="${GYM_VERIFY_HOST}" \
  +swe_pivot_single_step_tool_use_with_argument_comparison_resources_server.resources_servers.single_step_tool_use_with_argument_comparison.port="${GYM_VERIFY_PORT}" \
  +swe_pivot_single_step_tool_use_with_argument_comparison_resources_server.resources_servers.single_step_tool_use_with_argument_comparison.num_processes="${SWE_NUM_PROCESSES:-8}"
