# A800 上 vLLM FlashInfer Sampler 启动失败复盘

## 1. 文档目的

本文记录在 NVIDIA A800 80GB 上启动 vLLM 推理 Qwen3.5 系列模型时遇到的一次兼容性故障，并说明最终采用的解决方案：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

该配置只禁用 FlashInfer 的 token sampler，使 vLLM 回退到原生 PyTorch sampler；它不禁用 FlashAttention，不改变模型权重，也不关闭张量并行。

本文重点回答以下问题：

1. 为什么模型权重已经加载完成，vLLM 仍然在 warmup 阶段退出；
2. 为什么问题不是 A800 硬件不支持 Qwen3.5；
3. 禁用 FlashInfer sampler 具体改变了什么；
4. 对生成质量、随机性、吞吐和显存有什么影响；
5. 如何在新服务器上复现检查并采用标准启动方式。

## 2. 故障摘要

### 2.1 表面现象

启动流程能够完成以下步骤：

- 识别 GPU；
- 创建 tensor parallel worker；
- 加载模型配置；
- 加载大部分或全部模型权重；
- 分配部分 KV cache 和运行时显存。

但在 vLLM warmup 阶段，FlashInfer 尝试即时编译 top-k / top-p sampling CUDA 扩展，编译进程调用了系统的 `/usr/bin/nvcc`，随后退出。

核心错误为：

```text
CUDA versions below 12 are not supported.
```

上层最终只显示：

```text
Engine core initialization failed
```

因此，单看最后一行容易误判为模型架构、显存或 A800 不受支持。真正的首个错误出现在更早的 FlashInfer JIT 编译日志中。

### 2.2 最终结论

故障并非由以下原因引起：

- A800 不能运行 Qwen3.5；
- A800 不支持 BF16；
- 模型权重损坏；
- tensor parallel 无法工作；
- FlashAttention 2 不支持 A800。

真正原因是同一个进程中存在三套容易被混淆的 CUDA 版本信息：

| 层级 | 作用 | 当时状态 |
|---|---|---|
| NVIDIA Driver | 提供 GPU 驱动和 CUDA Driver API | GPU 可正常识别和分配显存 |
| PyTorch CUDA Runtime | PyTorch wheel 随包使用的运行时 | CUDA 12.9 构建 |
| 系统 `nvcc` | 编译本地 CUDA 扩展 | `/usr/bin/nvcc` 低于 CUDA 12 |

当时保留下来的具体版本组合为：

```text
GPU                         NVIDIA A800 80GB
NVIDIA Driver               535.183.06
PyTorch / vLLM CUDA Runtime 12.9
系统 /usr/bin/nvcc          11.5
FlashInfer                  0.6.12
vLLM                        0.24.0
```

vLLM 在失败前已经正确解析出：

```text
Resolved architecture: Qwen3_5ForConditionalGeneration
```

这条日志很重要：它证明模型架构识别已经通过，问题发生在后续 sampler warmup，而不是“vLLM 不认识 Qwen3.5”。

模型前向可以使用 PyTorch wheel 的 CUDA runtime，但 FlashInfer 的某些 sampler kernel 需要在首次运行时 JIT 编译。JIT 不会自动使用 `torch.version.cuda` 对应的编译器，而是从 `PATH`、`CUDA_HOME` 等位置找到系统 `nvcc`。因此出现了：

```text
PyTorch runtime 是 CUDA 12.9
        +
JIT 找到的 nvcc 低于 CUDA 12
        ↓
FlashInfer 0.6.x / CCCL 拒绝编译
```

## 3. 为什么权重加载完成后才失败

vLLM 启动并不只有“加载权重”一个阶段。简化流程为：

```text
解析模型配置
  ↓
创建 GPU worker 和 tensor parallel 进程组
  ↓
加载模型权重
  ↓
分析可用显存并创建 KV cache
  ↓
编译或准备运行时 kernel
  ↓
CUDA Graph / sampler warmup
  ↓
服务就绪
```

当时失败发生在 sampler warmup，而不是模型权重加载阶段。因此日志里先出现“Loading weights 100%”，不能证明整个服务已经启动成功。

可靠的“服务已就绪”判据应是：

- API 服务开始监听端口，且 `/v1/models` 返回成功；或
- 离线 vLLM engine 完成第一次真实 `generate` / `encode`；
- worker 进程仍存活；
- 输出文件开始持续增加。

仅看到显存占用、权重加载完成或者 CUDA Graph 开始创建，都不足以判断启动成功。

## 4. FlashInfer sampler 是什么

### 4.1 sampler 在生成链路中的位置

生成模型每一步会输出整个词表的 logits。随后需要根据生成配置选择下一个 token：

```text
Qwen 前向计算
  ↓
输出 vocab logits
  ↓
temperature 处理
  ↓
top-k / top-p / min-p 等过滤
  ↓
从保留的概率分布中采样
  ↓
得到下一个 token
```

FlashInfer sampler 负责的是后半段的过滤和抽样 kernel。它不是 Qwen 模型主体，也不是注意力计算模块。

### 4.2 FlashInfer sampler 的主要作用

FlashInfer 为采样阶段提供经过优化的 CUDA kernel，目标包括：

- 减少 top-k / top-p 筛选的中间张量；
- 减少 GPU kernel launch 次数；
- 提高大 batch、大词表和高并发生成时的 token 选择吞吐；
- 降低部分采样操作的临时显存和同步开销。

对于长文本生成，完整耗时通常由以下部分组成：

1. 模型 attention 和 MLP 前向；
2. KV cache 读写；
3. logits 计算；
4. sampler 选择 token。

FlashInfer sampler 只优化第 4 项。

## 5. 禁用 FlashInfer sampler 后发生了什么

设置：

```bash
VLLM_USE_FLASHINFER_SAMPLER=0
```

会让 vLLM 不再走 FlashInfer 的 top-k / top-p sampler，实现回退到 vLLM 支持的原生 PyTorch sampling 路径。

### 5.1 被禁用的内容

- FlashInfer top-k / top-p sampling kernel；
- 该 sampler 首次运行时的本地 JIT 编译；
- sampler 对应的 FlashInfer CUDA 扩展缓存路径。

### 5.2 没有被禁用的内容

- Qwen3.5 模型主体；
- 模型权重和 LoRA；
- BF16 前向；
- tensor parallel；
- KV cache；
- CUDA Graph 的其他部分；
- FlashAttention 2 attention backend；
- tokenizer 和 chat template；
- temperature、top-k、top-p、min-p 等生成参数本身；
- thinking / nonthinking Prompt 逻辑。

尤其需要注意：

```text
禁用 FlashInfer sampler
不等于
禁用 FlashInfer attention backend
也不等于
禁用 FlashAttention
```

当时日志仍能看到 vLLM 使用 FlashAttention 2。模型中占主要计算量的 attention 和 MLP 并没有因此回退到 CPU。

### 5.3 两种 sampler 的直接对比

| 维度 | FlashInfer sampler | 原生 PyTorch sampler |
|---|---|---|
| 实现方式 | 融合、优化的 CUDA sampling kernel | vLLM/PyTorch 通用算子路径 |
| top-k / top-p 语义 | 保留 | 保留 |
| temperature 等生成配置 | 保留 | 保留 |
| 首次运行 JIT | 当前版本可能触发并依赖可用 `nvcc` | 不依赖本次失败的 FlashInfer sampling JIT |
| 大并发采样吞吐 | 通常更优 | 可能略低 |
| 模型前向 | 不负责 | 不负责 |
| attention / MLP | 不负责 | 不负责 |
| 模型权重和 LoRA | 不改变 | 不改变 |
| 输出是否逐 token 相同 | 不保证 | 不保证 |
| 当前 A800 环境可用性 | 因 `nvcc 11.5` 编译失败而不可用 | 已验证可用 |

因此，回退不是把 GPU 推理改成 CPU 推理，也不是关闭所有 FlashInfer/FlashAttention 能力，只是把“根据 logits 选下一个 token”这一步换成通用实现。

## 6. 对结果质量和随机性的影响

### 6.1 理论采样分布不变

给定相同 logits 和相同配置，原生 PyTorch sampler 与 FlashInfer sampler执行的是相同类型的概率变换：

- temperature；
- top-k；
- top-p；
- min-p；
- multinomial sampling 或 greedy selection。

因此，这个开关不是“改变采样策略”，而是“替换采样策略的底层 kernel 实现”。生成配置仍由原来的 JSON 或命令行参数决定。

### 6.2 不保证逐 token 完全相同

在随机采样模式下，即使 seed、temperature、top-p 和 top-k 相同，也不应要求两种 sampler 输出逐 token 完全一致。原因包括：

- 浮点归约顺序不同；
- kernel 内部排序和阈值边界处理存在微小数值差异；
- GPU 随机数消费顺序可能不同；
- 并发调度和动态 batch 可能改变请求执行顺序。

这些差异可能让某一步抽到不同 token，随后整段生成发生分叉。但从统计意义上，两条实现应遵循同一目标采样配置。

对于当前教师 pair 标注，正确验收方式不是比较两次输出是否逐字一致，而是比较：

- JSON 解析成功率；
- A/B 方向一致率；
- soft target 的平均绝对差；
- 正反序位置偏差；
- 多次采样的票数分布。

### 6.3 temperature=0 的情况

如果使用纯 greedy decoding，sampler 实现差异通常更小，因为目标是选择最大 logit token。但极端情况下如果多个 token 的 logit 非常接近，浮点差异仍可能改变并列边界结果。

本项目的 Qwen3-32B pair 教师为了获得 soft vote 概率，并非统一使用 temperature=0，因此不能用“同 seed 必须完全复现”作为验收门槛。

### 6.4 生成式 teacher 与 pooling 难度模型要分开判断

本项目存在两类 vLLM 工作负载：

1. Qwen3-32B 教师生成 A/B 判定文本；
2. Qwen3.5 难度模型使用 pooling/encode 得到题目表示，再经过外置任务头得到难度分数和辅助特征。

对第一类任务，sampler 位于真实生成链路中，回退后可能改变随机采样的具体 token，并带来少量采样性能差异。

对第二类任务，如果实际调用的是 pooling/encode 而不是 `generate`，请求本身不需要从词表分布采样 token。此时 sampler 不参与题目表示、difficulty head 或 auxiliary head 的数值计算：

```text
题目文本
  ↓
Qwen backbone 前向
  ↓
pooling hidden state
  ↓
difficulty / auxiliary heads
```

所以对规范的 pooling 难度推理而言，禁用 FlashInfer sampler 原则上不会改变难度标量或辅助特征 logits，也基本不会影响单题打分吞吐。设置该变量的意义主要是避免某些 vLLM 版本在 engine 初始化或 warmup 时仍准备可选 sampler，从而在真正执行 pooling 请求前就启动失败。具体版本是否初始化 sampler，应以启动日志为准。

## 7. 对性能的影响

### 7.1 可能变慢的部分

原生 PyTorch sampler 通常需要更多通用算子和中间操作。在以下工作负载中，性能差异可能更明显：

- batch 很大；
- 并发请求很多；
- 输出很短，模型前向占比下降；
- 词表很大；
- top-k / top-p 过滤配置复杂；
- 单卡模型较小，模型前向本身非常快。

### 7.2 当前项目的实际影响判断

当前 pair teacher 使用 32B BF16 模型，两张 A800 做 tensor parallel。模型前向、长输入 prefill 和 thinking token 生成占据主要耗时，sampler 不是主要瓶颈。因此禁用 FlashInfer sampler 的预期影响是：

- 启动稳定性显著提高；
- 可能损失少量 decode 吞吐；
- 不改变标签定义；
- 不影响 FlashAttention 2；
- 相比无法启动，性能退让是合理的工程选择。

不能把这个判断推广到所有服务。若未来部署高并发、短输出的在线接口，需要使用同一批请求实测以下两组：

```text
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_FLASHINFER_SAMPLER=0
```

至少记录：

- time to first token；
- output tokens/s；
- requests/s；
- P50 / P95 / P99 延迟；
- 峰值显存；
- 解析成功率和输出分布。

在当前服务器 `nvcc` 未修复前，无法完成公平的 FlashInfer-on 对照，因为 on 路径不能通过 warmup。

## 8. 标准解决方案

### 8.1 推荐方案：回退到原生 PyTorch sampler

必须在导入 vLLM 或启动 worker 之前设置环境变量：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

离线 Python engine：

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  python scripts/run_local_pairwise_teacher.py \
    --config configs/qwen3_32b_pairwise_teacher_thinking_1024.json \
    --model-path /path/to/Qwen3-32B \
    --pairs /path/to/pairs.jsonl \
    --raw-votes-output /path/to/raw_votes.jsonl \
    --manifest /path/to/teacher.manifest.json \
  > /path/to/run.log 2>&1 &
```

OpenAI 兼容服务：

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --tokenizer /path/to/model \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 4096 \
    --host 127.0.0.1 \
    --port 8002 \
  > /path/to/vllm.log 2>&1 &
```

项目当前相关启动脚本已经主动设置该变量，教师 Python 入口也包含默认回退保护。

### 8.2 当时 Qwen3.5-9B 的已验证启动方式

当时没有升级驱动，也没有更换模型，而是在独立的、已验证运行环境中执行：

```bash
conda activate vime-runtime

export MODEL_PATH=/home/zhangyonglin/models/models/Qwen--Qwen3.5-9B/snapshots/master
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=7 vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3.5-9B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --language-model-only \
  --trust-remote-code \
  --enforce-eager
```

修复后日志应能观察到类似信息：

```text
FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0
Resolved architecture: Qwen3_5ForConditionalGeneration
```

随后模型能够正常加载全部 safetensors 分片并进入可服务状态。`--enforce-eager` 是当时为了优先验证兼容性和减少 CUDA Graph 相关变量采用的保守配置，并不是 FlashInfer sampler 回退的必要条件。

### 8.3 可选方案：修复系统 CUDA toolkit

另一种方案是安装与 PyTorch / FlashInfer 要求匹配的 CUDA 12.x toolkit，并确保 JIT 使用正确的编译器：

```bash
export CUDA_HOME=/path/to/cuda-12.x
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

然后确认：

```bash
which nvcc
nvcc --version
```

这条路线理论上可以重新启用 FlashInfer sampler，但风险更高：

- 可能破坏当前已经验证的 PyTorch、vLLM 和 CUDA 依赖组合；
- 编译器版本正确不代表所有动态库路径都正确；
- 不同服务器驱动版本可能不足以承载新 runtime；
- 重新编译扩展会引入新的可复现性变量。

当前项目优先保证标注任务稳定运行，因此不建议为了 sampler 的有限性能收益修改已验证环境。

### 8.4 不推荐方案：直接改 FlashInfer 源码绕过检查

错误信息可能建议定义某个宏以忽略旧 CUDA 检查，但这只会跳过版本保护，不保证旧编译器真正支持依赖的 CUDA / CCCL 特性。除非能够完成完整编译、数值和压力测试，否则不应在生产标注环境采用。

## 9. 启动前的版本审计

在新服务器上不要只看 `nvidia-smi`。应同时记录以下信息：

```bash
nvidia-smi

which nvcc || true
nvcc --version || true

python - <<'PY'
import importlib.metadata as md
import torch

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

for package in ("vllm", "flashinfer-python"):
    try:
        print(package + ":", md.version(package))
    except md.PackageNotFoundError:
        print(package + ": NOT INSTALLED")
PY
```

需要分别理解：

- `nvidia-smi` 显示的是驱动能支持到的 CUDA API 范围，不是本机 `nvcc` 版本；
- `torch.version.cuda` 是 PyTorch wheel 的构建 runtime 版本，不是系统编译器版本；
- `nvcc --version` 才是本地 CUDA 扩展 JIT 编译时最可能使用的编译器版本。

## 10. 修复后的验收

### 10.1 进程和 GPU

```bash
ps -ef | grep -E 'vllm|run_local_pairwise_teacher' | grep -v grep
nvidia-smi -i 0,1
```

不能只看显存。还应观察 GPU-Util 是否随真实请求升高，以及主 worker 是否持续存活。

### 10.2 日志

```bash
grep -E \
  'FlashInfer|nvcc|ninja|CUDA versions below|Engine core initialization failed|CUDA out of memory' \
  /path/to/run.log
```

修复后不应再出现 FlashInfer sampling JIT 的 `ninja` 编译失败。

### 10.3 API smoke test

```bash
curl -fsS http://127.0.0.1:8002/v1/models | python -m json.tool
```

再发送一条最小生成请求，确认服务不仅监听端口，而且能够完成一次前向和采样。

### 10.4 离线 teacher smoke test

至少检查：

- 输入 pair 数；
- raw vote 行数是否增长；
- JSON parse success rate；
- 正反序是否都有有效票；
- manifest 是否生成；
- 服务退出码是否为 0。

## 11. 修复后出现过的第二类问题

禁用 FlashInfer sampler 解决的是 JIT 编译失败。之后曾出现另一项独立问题：第一轮生成时显存不足。

当时配置组合包括：

- `gpu_memory_utilization=0.90`；
- 默认 `max_num_batched_tokens=16384`；
- 外层 batch 过大；
- `n=3` 多采样进一步展开请求；
- JSON 配置被 argparse 默认值意外覆盖。

这会导致 KV cache 预占过高、prefill activation 和 CUDA Graph 缺少余量。后续调整为：

```yaml
gpu_memory_utilization: 0.82
max_num_batched_tokens: 4096
max_num_seqs: 32
nonthinking_prompt_batch_size: 8
thinking_prompt_batch_size: 4
```

因此诊断顺序应为：

```text
FlashInfer / nvcc 编译报错
  → 禁用 FlashInfer sampler

CUDA out of memory
  → 调低显存预占、batch、batched tokens 和并发序列
```

不能用降低 batch 的办法解决编译器版本错误，也不能认为禁用 sampler 会自动解决所有 OOM。

## 12. 更早出现过的第三类问题：FLA 安装污染运行环境

在为 Qwen3.5 补充 GDN/FLA 相关依赖时，还发生过一次与 sampler 不同的环境问题。直接运行：

```bash
pip install "flash-linear-attention[cuda]"
```

会让 pip 根据当时最新依赖重新求解整个 CUDA 栈。历史安装日志显示，它试图拉取：

```text
torch 2.13.0
CUDA 13.0 相关依赖
triton 3.7.1
```

而服务器驱动为 `535.183.06`，无法支持这组 CUDA 13 runtime，随后可能出现：

```text
The NVIDIA driver on your system is too old
```

这不是 Qwen3.5 架构错误，也不是 FlashInfer sampler JIT 的同一个问题，而是 pip 在安装可选 CUDA 扩展时替换了原本可用的 Torch/CUDA 组合。

### 12.1 当时采用的环境策略

```text
1. 不在 QuRater 训练环境中直接升级整套 CUDA 依赖；
2. 单独建立 vime-runtime / vLLM serving 环境；
3. 固定已经验证过的 Torch、vLLM、Transformers 版本；
4. FLA 类包必要时使用 --no-deps，禁止 pip 自动替换 Torch；
5. causal-conv1d 使用与既有 Torch/CUDA ABI 匹配的预编译 wheel；
6. 不为了一个 Python 扩展盲目升级 535 驱动或安装系统 CUDA 12.9/13；
7. 安装前后保存 conda explicit 和 pip freeze，并执行 pip check 与真实 smoke test。
```

使用 `--no-deps` 的前提是人工确认依赖已经满足。它是防止环境被替换的手段，不是忽略依赖兼容性的许可。

### 12.2 三类故障的边界

| 故障 | 发生阶段 | 典型错误 | 对应处理 |
|---|---|---|---|
| FlashInfer sampler JIT 不兼容 | vLLM warmup | `Ninja build failed`、`FlashInfer_sampling_ninja_build_failed` | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| 批量推理 OOM | 服务已启动后的真实 generate | `CUDA out of memory` | 降低显存预占、batch、batched tokens、并发序列 |
| FLA 安装污染环境 | pip 安装或首次 import/运行 | 拉取 Torch/CUDA 13、驱动过旧 | 隔离环境、锁版本、匹配 wheel、谨慎使用 `--no-deps` |

三个问题可能依次出现，但根因和解决办法完全不同。排障时必须定位“第一条真正的异常”，不能只看最后的 `Engine core initialization failed` 或 `driver too old`。

## 13. 对 Qwen3.5 难度模型部署的额外说明

如果部署的是本项目训练后的 Qwen3.5-4B 难度模型，还需要区分“vLLM 能加载 backbone”和“能够输出最终难度”两件事。

训练 checkpoint 包含：

```text
Qwen3.5 backbone
+ LoRA adapter
+ scalar difficulty head
+ optional auxiliary heads
```

普通 `vllm serve` 只负责 Qwen backbone 和支持的 LoRA，不会自动读取项目自定义的 `pairwise_head.pt`。因此生产难度推理应使用：

```text
vLLM pooling runner
  ↓
加载 LoRA
  ↓
返回 LAST hidden state
  ↓
外置 PyTorch difficulty / auxiliary heads
  ↓
calibration 后处理
```

这与 FlashInfer sampler 故障是两个不同层面的问题：

- FlashInfer sampler 问题决定 vLLM 生成服务能否完成 warmup；
- 外置任务头问题决定加载训练 checkpoint 后能否得到正确的难度分数。

即使 FlashInfer sampler 已修复，也不能直接把普通生成接口的文本输出当作难度模型分数。

## 14. 最终工程决策

当前 A800 环境采用以下冻结策略：

```yaml
hardware:
  gpu: NVIDIA A800 80GB

runtime_policy:
  use_verified_project_vllm_environment: true
  modify_existing_cuda_stack: false
  vllm_use_flashinfer_sampler: false
  attention_backend: FlashAttention_2
  qwen35_gdn_fla_kernels: retained_when_environment_supports_them
  tensor_parallel: enabled_when_required

reason:
  - system_nvcc_is_older_than_flashinfer_jit_requirement
  - native_pytorch_sampler_is_supported_and_stable
  - sampler_is_not_the_primary_compute_bottleneck_for_current_32B_teacher_workload
```

最终结论是：**这次故障不是 A800 无法推理 Qwen3.5，而是 FlashInfer sampler 的 JIT 编译器版本与运行环境错位。禁用 FlashInfer sampler 后，vLLM 只在 token 采样实现上回退到 PyTorch，模型主体、FlashAttention、BF16 和多卡推理能力保持不变。**

## 15. 证据位置

项目内已记录的相关证据包括：

- `docs/experiment_log.md`：FlashInfer warmup 失败、根因和修复记录；
- `docs/pairwise_v3.md`：A800 双卡 teacher 配置和 sampler 回退说明；
- `scripts/run_local_pairwise_teacher.py`：运行前设置 `VLLM_USE_FLASHINFER_SAMPLER=0`；
- `scripts/server_run_cascade_production.sh`：生产 cascade 启动时设置相同变量；
- `tests/test_teacher_reasoning_experiment.py`：验证默认禁用 FlashInfer sampler。
