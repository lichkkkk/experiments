# LLM Technical Topics — Personal Study Notes

> Compiled from past discussions. Topics span RL theory, architecture, post-training, inference, and research frontiers.

-----

## Table of Contents

1. [Training & Alignment Algorithms](#1-training--alignment-algorithms)
1. [LLM Architecture](#2-llm-architecture)
1. [Post-Training & Evaluation](#3-post-training--evaluation)
1. [Inference & Serving](#4-inference--serving)
1. [Research Frontiers](#5-research-frontiers)
1. [Practical & Tooling](#6-practical--tooling)

-----

## 1. Training & Alignment Algorithms

### 1.1 REINFORCE → PPO → GRPO Evolution

**REINFORCE (vanilla policy gradient)**

- Update rule: `θ ← θ + α · ∇log π(a|s) · G_t`
- Problem: Very high variance, especially on long sequences
- Baseline trick: Subtract a baseline `b(s)` from the return — doesn't introduce bias (by linearity of expectation), but reduces variance significantly

**A3C / A2C (Actor-Critic)**

- Introduce a Critic (value network) as learned baseline
- Inner loop: actor collects experience; outer loop: critic updates `V(s)` via TD error
- A2C = synchronous version of A3C

**TRPO (Trust Region Policy Optimization)**

- Key insight: don't let policy update too far — constrains the KL divergence between old and new policy
- Optimization: `max E[r(θ) · A]` subject to `KL(π_old || π_new) ≤ δ`
- Computationally expensive due to Fisher information matrix inversion

**PPO (Proximal Policy Optimization)**

- Approximates TRPO with a simpler clipped objective:
  `L_CLIP = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]`
- `r(θ) = π_new(a|s) / π_old(a|s)` — importance sampling ratio
- Per-token advantages: different advantage values for each token position
- Requires training a separate value network (critic) — expensive

**GRPO (Group Relative Policy Optimization)**

- DeepSeek's contribution — eliminates value network
- Sample G completions for the same prompt, compute sequence-level rewards
- Advantage for each sequence: `A_i = (R_i - mean(R)) / std(R)` — group-normalized
- All tokens in sequence i get the same advantage value (sequence-level, not per-token)
- Key insight: regresses toward REINFORCE+baseline, but with group comparison as baseline
- Trade-off: higher variance than PPO per-token, but avoids bias from poorly trained critic

**DPO (Direct Preference Optimization)**

- Bypasses RL entirely — directly optimizes policy from preference pairs
- Reparameterizes the reward in terms of the policy itself:
  `r(x, y) = β · log(π(y|x) / π_ref(y|x))`
- Loss: binary cross-entropy on preferred vs rejected responses
- No reward model needed; much simpler to train

-----

### 1.2 Multi-Armed Bandits

**Core problem**: Choose between K arms to maximize cumulative reward — exploration vs exploitation

- ε-greedy: exploit best arm with probability 1-ε, explore randomly with ε
- UCB (Upper Confidence Bound): `UCB = Q(a) + c·√(ln t / N(a))` — optimistic exploration
- **Thompson Sampling**: maintain Beta(α, β) prior for each arm; sample θ_i ~ Beta; pick argmax
  - Beta distribution used because it's the conjugate prior to Binomial — posterior stays Beta after Bernoulli observations
  - α = successes + 1, β = failures + 1 → posterior updated analytically

**Connection to LLMs (Contextual Bandits)**

- Each prompt is a context; each response is an arm
- RLHF reward model ≈ bandit reward signal
- Contextual bandits = bandits with state (the prompt) — closer to full RL but without temporal credit assignment

-----

### 1.3 Policy Iteration vs Value Iteration

|Property   |Policy Iteration                    |Value Iteration                 |
|-----------|------------------------------------|--------------------------------|
|Steps      |Alternates policy eval + improvement|Combines into one Bellman backup|
|Convergence|Fewer iterations, each expensive    |More iterations, each cheap     |
|Operator   |Policy Bellman operator             |Optimal Bellman operator        |
|Policy     |Explicit; updated each round        |Implicit; extracted at the end  |

**Policy Gradient Theorem**: `∇J(θ) = E[∇log π(a|s) · Q(s,a)]`

- Bridges value-based and policy-based methods
- Actor-Critic = policy gradient with learned Q as baseline

-----

### 1.4 Inner/Outer Loop in RL and LLM Agents

**Traditional Meta-RL (MAML)**

- Outer loop: optimize for fast adaptability across tasks (meta-gradient)
- Inner loop: few gradient steps on a specific task

**LLM Agents**

- Inner loop: in-context reasoning (ReAct, chain-of-thought) — no parameter updates
- Outer loop: prompt optimization, memory updates, tool strategy
- Reflexion: outer loop uses verbal self-reflection to update behavior
- Key difference: LLM inner loops are parameter-free and deployment-time

-----

### 1.5 ScaleRL — Predictable RL Training

Key contributions from the ScaleRL paper (2025):

- **S-curve scaling law**: `perf = A · (1 - exp(-B · compute))`
  - A = performance ceiling, B = training efficiency
- **PipelineRL**: async actor-critic with decoupled rollout and update
- **CISPO loss**: importance-sampling correction for off-policy data
- **FP32 precision**: critical for RL stability (FP16 causes gradient explosion)
- **Adaptive data filtering**: removes outlier rollouts before update
- Beat GRPO, DAPO, Magistral baselines; scaling behavior validated across model sizes

-----

## 2. LLM Architecture

### 2.1 Normalization Techniques

**Batch Normalization**

- Normalizes over (N, H, W) for each channel C independently
- Problem in Transformers: batch dependency, variable sequence lengths

**Layer Normalization (LayerNorm)**

- Normalizes over feature dimension D for each token independently
- `y = (x - μ) / σ · γ + β`
- Key point: operates *within* a single token across its D features — not across tokens
- Post-LN (original Transformer): LN after residual add → training instability at scale
- Pre-LN (modern default): LN before sublayer → stable gradients, now standard

**RMSNorm**

- `y = x / RMS(x) · γ`, where `RMS(x) = √(mean(x²))`
- Drops mean centering (β parameter removed) — faster, slightly less expressive
- Used in LLaMA, Mistral, Gemma, DeepSeek

**DeepNorm**

- Scales residual connections: `x_new = α · x + sublayer(x)`
- With large α (e.g., 0.87·N^0.25), Pre-LN stability is maintained for very deep networks

**Does normalization affect final performance, not just training?**

- Yes — normalization changes the model's hypothesis space through scaling invariance
- RMSNorm inductive bias: scale-invariant representations, different loss landscape geometry

-----

### 2.2 Position Encoding

**Sinusoidal (original Transformer)**

- `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)`
- Applied once at input layer, propagates through residual connections
- No learnable params; can generalize to unseen lengths theoretically

**Learnable Embeddings (BERT, GPT)**

- Position embeddings as lookup table — `P ∈ R^{max_len × d}`
- Cannot generalize beyond max_len

**Relative Position Encoding (T5)**

- Adds learned relative position bias to attention logits
- `a_{ij} = q_i · k_j + b(i-j)` — encodes distance, not absolute position

**RoPE (Rotary Position Encoding)** ← current standard

- Rotates Q and K vectors in 2D subspaces by angle `m · θ_i`
- `θ_i = base^(-2i/d)`, where base = 10000 (or 500000 for long context in LLaMA 3.1)
- Key property: `(R_m·q)^T (R_n·k) = q^T R_{m-n} k` — dot product encodes relative position naturally
- Applied at **every attention layer**, not just input — position remains explicit throughout the network
- Long-context extensions: YaRN, Position Interpolation, NTK-aware scaling

**RoPE variants (2025)**

- M-RoPE (Qwen2-VL): 3D rotation — time, height, width dimensions for multimodal
- TM-RoPE (Qwen3-Omni): angle budget (24 dims for time, 20 each for H/W)
- LLaMA 3.1: base frequency 500,000 for 128K context

-----

### 2.3 Non-Markovian Processes in RL

**Markov Property**: `P(s_{t+1} | s_t, s_{t-1}, ...) = P(s_{t+1} | s_t)`

**Non-Markovian examples**:

- Financial markets with momentum (price depends on trend history)
- NLP: word probability depends on full sentence context, not just last word
- Metal fatigue: failure probability accumulates over entire load history

**Handling in practice**: State augmentation — encode history into expanded state space

- Frame stacking in Atari (last 4 frames → pseudo-Markov)
- LSTMs/GRUs encode history in hidden state
- Transformer attention = full history access in context window = implicit state augmentation

-----

## 3. Post-Training & Evaluation

### 3.1 Classification Metrics for LLM Evaluation

**PR AUC advantages over ROC AUC for imbalanced datasets**

- ROC AUC accounts for TN → inflated for rare-positive cases (e.g., factual errors in a mostly-correct model)
- PR AUC focuses only on precision/recall — more sensitive to minority class performance
- Random classifier baseline: PR AUC ≈ class prevalence `p`; ROC AUC ≈ 0.5 always

**Threshold selection**

- F1-optimal: maximize `F1 = 2·P·R / (P+R)` — balances precision and recall
- Cost-sensitive: define cost matrix `C(FN) vs C(FP)` — in factuality, FN (missing errors) >> FP cost
- Pick operating threshold on the PR curve based on cost ratio

**Confidence Intervals**

- Bootstrap CI for PR AUC: resample N times, compute AUC distribution
- Wilson interval for fixed-threshold precision/recall
- Critical for A/B decisions in production — never rely on point estimates alone

-----

### 3.2 AutoRater / LLM-as-Judge

**Core idea**: Use a stronger LLM to rate outputs of a target LLM

|Mode           |Description                         |Tradeoff                         |
|---------------|------------------------------------|---------------------------------|
|Pairwise       |"Which response is better — A or B?"|Reduces anchoring; more expensive|
|Pointwise      |"Rate on 1-5"                       |Granular; higher variance        |
|Reference-based|Compare to gold answer              |Requires gold labels             |

**Key biases**:

- Position bias: judges favor first response or longer responses
- Self-preference: models rate their own outputs higher
- Verbosity bias: length conflated with quality
- Mitigations: swap positions, CoT before rating, calibrate on human labels

**At scale**: Train domain-specific judge on human prefs; use for regression testing; ensemble multiple judges

-----

### 3.3 LLM Agent Evaluation

**Key benchmarks**:

|Capability           |Benchmark    |
|---------------------|-------------|
|Software engineering |SWE-bench    |
|General assistants   |GAIA         |
|Web navigation       |WebArena     |
|Function/tool calling|Berkeley BFCL|

**CLASSic framework**: Cost · Latency · Accuracy · Stability · Security

**Production observability**: Datadog LLM Observability, Langfuse, Arize Phoenix

- Metrics: latency P50/P99, token usage, groundedness, error rates

**Key trends (2024-2025)**:

- Evaluation-Driven Development (EDD) — eval as first-class engineering concern
- Static → dynamic/adversarial benchmarks
- Domain-specific agents outperform general foundation models in production
- Tiered evaluation: unit tests → regression → human spot-check

-----

### 3.4 LLM Instruction Following (2025 Research)

**State of the field**: Even frontier models (GPT-4o, Claude 3.5) achieve <50% on hard multi-constraint benchmarks

**Failure modes**:

- Multi-constraint: satisfying A while satisfying B and C simultaneously
- Long context degradation: instruction following quality drops with context length
- Format adherence: JSON schema, length constraints, structured outputs

**Key 2025 benchmarks**: IFBench, EIFBENCH, StructFlowBench

**Training innovations beyond SFT/RLHF**:

- Process Reward Models (PRM): reward intermediate reasoning steps
- Constitutional AI variants: automated critique + revision loops
- Instruction hierarchy: differentiated signal weighting for system/user/assistant turns

-----

## 4. Inference & Serving

### 4.1 KV Cache

**What it is**: Cache Key/Value tensors from processed tokens to avoid recomputation during decode

- New token at position t+1: compute Q_{t+1} only; attend to cached K_{1..t}, V_{1..t}
- Memory grows linearly with sequence length × layers × heads × d_head × 2

**Operations**:

|Operation            |KV Cache Effect                                                    |
|---------------------|-------------------------------------------------------------------|
|New user turn        |Append new tokens; prefill only new portion                        |
|Model generates token|Append one K/V entry per layer                                     |
|User interruption    |Truncate assistant tail; append new user tokens                    |
|Long conversation    |Cache grows; eviction strategies: sliding window, H2O, StreamingLLM|

-----

### 4.2 LLM Live API Internals

**Streaming output** (straightforward):

- Autoregressive decode → one token per forward pass → flush via SSE/WebSocket immediately

**Simultaneous input handling** (hard part):

|Strategy                    |Mechanism                                                                |
|----------------------------|-------------------------------------------------------------------------|
|Turn-based (most common)    |VAD detects end-of-turn → full prefill                                   |
|Interruption handling       |VAD detects user speech → cancel generation → KV truncate → re-prefill   |
|Streaming prefill (research)|Chunk-wise prefill concurrent with decode; requires disaggregated serving|

**Disaggregated prefill/decode serving**:

- Prefill is compute-bound; decode is memory-bandwidth-bound
- Run on separate GPU pools → no resource contention
- Enables true parallel: decode while accepting new input

**Full pipeline**:

```
Audio → ASR → Token stream
                    ↓
            Scheduling Layer (VAD + interrupt logic)
                    ↓
            KV Cache Manager
                    ↓
        [Prefill GPUs] | [Decode GPUs]
                    ↓
            Token stream → TTS → Audio output
```

**Pricing dynamics**:

- Audio tokens ~6-8x more expensive than text tokens (OpenAI Realtime)
- Context accumulates across turns → cost grows ~quadratically per session
- Cost strategies: use text-in/audio-out, truncate context aggressively, tune VAD sensitivity

-----

## 5. Research Frontiers

### 5.1 LLMs as Scientific Discovery Agents

**Key systems**:

|System                |Capability                                               |
|----------------------|---------------------------------------------------------|
|Coscientist           |End-to-end chemistry; integrates lab robotics            |
|ChemCrow              |Tool-augmented LLM; validated novel chromophore discovery|
|AI Scientist (Sakana) |Generate + run + write + review ML papers; ~$15/paper    |
|Virtual Lab (Stanford)|Multi-agent drug discovery simulation                    |

**Capability vs discovery gap**:

- Complete 30-60% of complex research tasks automatically
- Only ~3 independently validated novel discoveries as of 2025
- Strong at optimizing known processes; weak at paradigm-shifting insights

**Key limitations**: Hallucination rate 8-20%, reproducibility issues, dual-use safety risks

**Conclusion**: Future is human-AI symbiosis — AI handles high-throughput experimentation; humans provide creative direction and judgment

-----

### 5.2 Monte Carlo Tree Search (MCTS)

**Four steps**:

1. **Selection**: traverse tree via UCB1 until unexplored leaf
1. **Expansion**: add child node
1. **Simulation**: rollout to terminal (random or policy-guided)
1. **Backpropagation**: update win/visit stats up the path

**AlphaGo integration**:

- Policy network: replaces random simulation → guides selection
- Value network: replaces full rollout → instant position evaluation
- MCTS provides search structure; neural nets provide intuition

**LLM applications**:

- Tree-of-Thought: MCTS over reasoning chain steps
- PRM (Process Reward Model) as value function for rollout evaluation
- Self-play + MCTS for math/coding problem solving

-----

## 6. Practical & Tooling

### 6.1 Python Environment Management with `uv`

- Rust-based unified Python package + env manager — significantly faster than pip/venv
- Key commands:

  ```bash
  uv python install 3.13      # download + manage Python interpreter
  uv venv --python 3.13       # create isolated virtual env
  uv pip install torch         # fast dependency resolution
  ```
- `.python-version` or `pyproject.toml` for project-level pinning
- PyTorch wheel ABI tags (`cp311`, `cp312`, `cp313`) must match Python version — Python 3.14 often lacks wheels

-----

### 6.2 RPC Authentication & End-User Credentials (EUC)

**gRPC authentication options**:

- JWT/OAuth 2.0 token: passed via metadata `authorization: Bearer <token>`
- mTLS: mutual certificate verification at transport layer
- API key: simple but no user identity propagation

**End-User Credential (EUC) propagation in multi-tier RPC**:

|Strategy           |Mechanism                                                                       |Risk                             |
|-------------------|--------------------------------------------------------------------------------|---------------------------------|
|Token propagation  |Pass original user JWT through call chain                                       |Wide blast radius if token leaked|
|Token exchange     |OAuth 2.0 delegation — service exchanges its token + user ID for delegated token|Safer; auditable                 |
|Context propagation|mTLS peer cert (service identity) + user context header (user identity)         |Clean separation                 |

**Google Stubby/gRPC pattern**: Maintains two separate identity channels — peer identity (service-to-service via mTLS) and end-user tickets (user identity forwarded via RPC metadata)

-----

*Last updated: February 2026*
