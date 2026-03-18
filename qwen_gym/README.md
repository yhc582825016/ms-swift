# qwen_gym

按能力拆分的 Gym + Swift 训练目录：

- `math/`
- `instruction_following/`
- `agent/`
- `swe_agent/`
- `workplace_assistant/`
- `calendar/`
- `common/`
- `data/`

## 1) 准备数据

```bash
export HF_ENDPOINT=https://hf-mirror.com
python /mnt/code/yehangcheng/ms-swift/qwen_gym/prepare_all_gym_data.py
```

如果只准备多步 agent 的 `workplace_assistant + calendar`：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python /mnt/code/yehangcheng/ms-swift/qwen_gym/prepare_workplace_calendar_data.py
```

## 2) 启动能力对应的 NeMo Gym verifier

例如 math：

```bash
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/math/start_nemo_gym.sh
```

## 3) 启动 Swift rollout（server 模式推荐）

```bash
NEMO_GYM_VERIFY_URL=http://127.0.0.1:18001/verify \
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/common/start_swift_rollout_gym.sh
```

## 4) 启动训练

例如 math：

```bash
bash /mnt/code/yehangcheng/ms-swift/qwen_gym/math/train_server.sh
```

instruction_following / agent / swe_agent / workplace_assistant / calendar 同理，使用对应目录下 `train_server.sh`。
