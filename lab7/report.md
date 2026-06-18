---
title: "人工智能实验报告"
subtitle: "中山大学计算机学院本科生实验报告"
author:
  - "课程名称：Artificial Intelligence"
  - "学号：24344064"
  - "姓名：廖海涛"
date: "2026-06-18"
---

# 实验题目

基于 LangGraph 的多智能体数学推理系统（MathNexus）

# 实验内容

## 1. 算法原理

### 1.1 系统概述

本实验设计并实现了一个名为 **MathNexus** 的多智能体数学推理系统。该系统采用 LangGraph 框架进行多智能体编排，集成了 Tree of Thoughts (ToT)、ReAct 和 Reflection 三种推理/规划方法，并实现了短期记忆和长期记忆两种记忆机制，以及四类工具供智能体调用。

系统参考了 Google DeepMind 的 AlphaProof / AlphaGeometry 的设计理念，采用"问题分类 → 求解 → 验证 → 反思 → 重试"的闭环架构，模拟人类解决数学问题的认知流程。

### 1.2 系统架构

系统包含 **5 个专门化智能体**，每个智能体有明确的角色定位和适配的提示词：

```
┌─────────────────────────────────────────────────────────────┐
│                     MathNexus Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入 ──► Orchestrator（先分类）                        │
│                 │                                           │
│         ┌───────┴────────┐                                  │
│         ▼                ▼                                  │
│    Prover (ToT)     Solver (ReAct)                          │
│    [证明题目专用]     [计算/选择/简答]                      │
│         │                │                                  │
│         └───────┬────────┘                                  │
│                 ▼                                           │
│           Validator（验证评分）                             │
│            │           │                                    │
│    若正确 ◄┘           └──► 若错误                          │
│         │                    │                              │
│         │              Reflector（反思改进）                │
│         │                    │                              │
│         │              [如未达最大尝试次数]                 │
│         │                    │                              │
│         │              则返回 Orchestrator                  │
│         │                    │                              │
│         └────────────────────┘                              │
│                 ▼                                           │
│           最终答案输出                                      │
│                                                             │
│  记忆系统：短期记忆（会话状态）+ 长期记忆（ChromaDB 向量库）│
│  工具集: Python REPL | SymPy | Web Search | Calculator      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 智能体设计

#### Agent 1: Orchestrator（编排者）

- **功能定位**: 问题分析、类型分类、求解策略推荐
- **提示词设计**: 要求输出结构化 JSON，包含 `problem_type`、`key_concepts`、`complexity`、`recommended_approach`、`rationale` 字段
- **分类类型**:
  - `computation`: 需要数值计算或方程求解的问题
  - `choice`: 选择题
  - `short_answer`: 简答题
  - `proof`: 需要严格数学证明的问题

#### Agent 2: Prover（证明者）— 使用 Tree of Thoughts

- **功能定位**: 处理证明题，采用 ToT 推理方法
- **ToT 实现原理**:
  1. **生成阶段 (Breadth)**: 并行生成 4 种不同的证明策略（如反证法、直接证明、归纳法、构造法）
  2. **评估阶段**: 对每种策略从 Correctness（正确性）、Completeness（完整性）、Feasibility（可行性）三个维度打分（1-10）
  3. **选择阶段**: 根据总分选出最优策略
  4. **展开阶段**: 将最优策略扩展为完整严格证明
- **提示词设计**: 包含数学定理陈述、证明策略概述、逐步推导、结论的标准格式要求

#### Agent 3: Solver（求解者）— 使用 ReAct

- **功能定位**: 处理计算题、选择题、简答题，采用 ReAct 推理方法
- **ReAct 实现原理**:
  1. **Thought（思考）**: 分析问题，确定需要计算什么
  2. **Action（行动）**: 调用工具（Python/SymPy/搜索/计算器）
  3. **Observation（观察）**: 检查工具输出
  4. 循环 Thought→Action→Observation，最多 8 步
  5. **Final Answer**: 提供完整解答
- **提示词设计**: 针对不同问题类型（computation/choice/short_answer）有专门的指令

#### Agent 4: Validator（验证者）

- **功能定位**: 验证解答的正确性、完整性，给出评分
- **验证维度**: 逻辑正确性、步骤完整性、边界条件检查、数值验证
- **提示词设计**: 要求输出结构化 JSON，包含 `is_correct`、`confidence`、`score`、`errors`、`warnings`、`strengths`、`suggestions` 字段

#### Agent 5: Reflector（反思者）— 使用 Reflection

- **功能定位**: 分析失败原因，提出改进方案
- **Reflection 实现原理**:
  1. 分析解答和验证反馈
  2. 识别错误模式（error patterns）和知识盲区（knowledge gaps）
  3. 生成具体改进方案（improvement plan）
  4. 提出备选策略（alternative approaches）
  5. 决定是否重试（should_retry）
- **提示词设计**: 要求输出结构化 JSON，包含 `root_cause`、`error_patterns`、`knowledge_gaps`、`improvement_plan`、`alternative_approaches`、`should_retry` 字段

### 1.4 记忆机制

#### 短期记忆 (Short-Term Memory)

- **实现方式**: LangGraph 内置的 `MemorySaver`（checkpointer）+ 自定义 `ShortTermMemory` 类
- **存储内容**:
  - 当前会话的对话消息缓冲区（最多 50 条）
  - 工具调用日志（tool_name、input、output）
  - 智能体间交互记录（from_agent、to_agent、message）
- **生命周期**: 随会话存在，每次新的问题求解开始时清空
- **用途**: 为智能体提供当前问题的上下文，支持多步推理和工具调用历史回顾

#### 长期记忆 (Long-Term Memory)

- **实现方式**: ChromaDB 向量数据库
- **存储机制**:
  - 将问题-解答对编码为文档嵌入向量（使用 all-MiniLM-L6-v2 模型）
  - 存储元数据：问题文本、解答文本、问题类型、成功状态、关键概念、时间戳
  - 使用问题的 MD5 哈希作为唯一标识符
- **检索机制**:
  - 在新问题求解前，对问题进行语义嵌入
  - 通过余弦相似度检索最相似的 3 个历史问题
  - 将检索到的相似问题及其解答作为上下文提供给智能体
- **持久化**: 数据持久化到本地磁盘，跨会话保持
- **用途**: 知识积累、相似问题参考、解题策略复用

### 1.5 工具集 (4 种工具)

#### Tool 1: Python REPL（python_repl）

- **功能**: 执行 Python 代码，捕获标准输出和标准错误
- **用途**: 数值计算、算法实现、结果验证
- **实现**: 使用 `contextlib.redirect_stdout` 捕获输出，带异常处理

#### Tool 2: SymPy 符号数学（sympy_math）

- **功能**: 符号数学计算
- **支持操作**: 解方程 (`solve`)、化简 (`simplify`)、展开 (`expand`)、因式分解 (`factor`)、微分 (`diff`)、积分 (`integrate`)、极限 (`limit`)、级数展开 (`series`)、矩阵运算 (`Matrix`)
- **用途**: 精确符号计算、公式推导、结果验证

#### Tool 3: Web Search（web_search）

- **功能**: 使用 DuckDuckGo 搜索引擎检索数学概念、定理和参考资料
- **用途**: 查找定理、检索数学概念、搜索类似问题
- **实现**: 使用 `duckduckgo_search` 库，返回前 5 条搜索结果

#### Tool 4: Calculator（calculator）

- **功能**: 安全地评估算术表达式
- **安全机制**: 使用 `ast.parse` 解析表达式，白名单机制限制可用函数
- **支持运算**: +, -, *, /, **, %, //, 以及 abs、round、sqrt、sin、cos、log 等数学函数

## 2. 关键代码展示

### 2.1 LangGraph 工作流定义

```python
def build_graph(model_name="deepseek-chat", temperature=0.3, ...):
    workflow = StateGraph(AgentState)

    # 添加节点：5 个智能体 + 最终输出节点
    workflow.add_node("orchestrator", orchestrate)
    workflow.add_node("prover", prove)         # ToT 推理
    workflow.add_node("solver", solve)          # ReAct 推理
    workflow.add_node("validator", validate)    # 验证评分
    workflow.add_node("reflector", reflect)     # Reflection 反思
    workflow.add_node("finalize", finalize)     # 最终输出

    workflow.set_entry_point("orchestrator")

    # 条件路由：根据问题类型分派到 Prover 或 Solver
    workflow.add_conditional_edges(
        "orchestrator", router,
        {"prover": "prover", "solver": "solver"},
    )

    workflow.add_edge("prover", "validator")
    workflow.add_edge("solver", "validator")

    # 验证后条件路由：正确→输出，错误→反思
    workflow.add_conditional_edges(
        "validator", should_retry,
        {"router": "reflector", "finalize": "finalize"},
    )

    # 反思后条件路由：重试→重新分类，放弃→输出
    workflow.add_conditional_edges(
        "reflector", should_retry,
        {"router": "orchestrator", "finalize": "finalize"},
    )

    workflow.add_edge("finalize", END)

    # 使用内存检查点实现短期记忆
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
```

### 2.2 Tree of Thoughts (ToT) 实现

```python
async def prover_node(state, llm):
    """使用 Tree of Thoughts 解决证明题"""
    # Phase 1: 生成多种证明策略（广度探索）
    num_strategies = 4
    gen_response = await llm.ainvoke([
        SystemMessage(content=PROVER_SYSTEM_PROMPT),
        HumanMessage(content=TOT_THOUGHT_GENERATION_PROMPT.format(
            problem=problem, num_strategies=num_strategies,
        )),
    ])
    strategies = parse_json(gen_response.content)["strategies"]

    # Phase 2: 评估每种策略（打分）
    eval_response = await llm.ainvoke([
        SystemMessage(content="You are a rigorous proof evaluator."),
        HumanMessage(content=TOT_EVALUATION_PROMPT.format(
            problem=problem, strategies_text=formatted,
        )),
    ])
    evaluations = parse_json(eval_response.content)["evaluations"]

    # Phase 3: 选择最优策略
    best_strategy = max(strategies, key=lambda s: score(s, evaluations))

    # Phase 4: 展开为完整证明
    expand_response = await llm_with_tools.ainvoke([
        SystemMessage(content=PROVER_SYSTEM_PROMPT),
        HumanMessage(content=TOT_EXPANSION_PROMPT.format(
            problem=problem, strategy_name=best_strategy["name"],
            outline=format_outline(best_strategy["outline"]),
        )),
    ])
    return {"solution": expand_response.content, ...}
```

### 2.3 ReAct 循环实现

```python
async def solver_node(state, llm, short_term_memory):
    """使用 ReAct 解决计算/选择/简答题"""
    max_react_steps = 8
    messages = [
        SystemMessage(content=SOLVER_SYSTEM_PROMPT),
        HumanMessage(content=problem),
    ]
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    for step in range(max_react_steps):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        # 如果没有工具调用，说明得出最终答案
        if not response.tool_calls:
            return {"solution": response.content, ...}

        # 执行工具调用 → 获取观察结果
        for tool_call in response.tool_calls:
            tool = TOOLS_BY_NAME.get(tool_call["name"])
            result = tool.invoke(tool_call["args"])  # Action
            messages.append(ToolMessage(content=str(result), ...))
            # result 即为 Observation，进入下一轮 Thought

    return {"solution": final_response.content, ...}
```

### 2.4 长期记忆 (ChromaDB) 实现

```python
class LongTermMemory:
    def __init__(self, persist_dir="./math_nexus_memory"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="math_problems"
        )

    def store(self, problem, solution, problem_type, success, key_concepts):
        """存储问题-解答对到向量数据库"""
        problem_id = hashlib.md5(problem.encode()).hexdigest()[:16]
        doc_text = f"Problem: {problem}\nType: {problem_type}\nSolution: {solution}"
        self.collection.upsert(
            ids=[problem_id],
            documents=[doc_text],
            metadatas=[{...}],
        )

    def retrieve(self, query, n_results=3):
        """语义检索相似的历史问题"""
        results = self.collection.query(
            query_texts=[query], n_results=n_results
        )
        return formatted_memories
```

## 3. 创新点 & 优化

### 3.1 三种推理方法的深度融合

本系统不是简单堆砌多种推理方法，而是将它们有机整合到一个闭环流程中：

- **ToT** 用于证明题（需要探索多种证明路径的场景）
- **ReAct** 用于计算题（需要工具辅助的交互式求解场景）
- **Reflection** 用于全局质量控制（对失败结果进行元认知分析并驱动重试）

### 3.2 双记忆架构

- **短期记忆**基于 LangGraph 的状态管理 + 自定义消息缓冲区，支持会话内的多步推理上下文保持
- **长期记忆**基于 ChromaDB 向量数据库，通过语义检索实现跨会话的知识复用
- 两种记忆各司其职，形成类似人类"工作记忆"和"长期知识"的认知架构

### 3.3 结构化提示词工程

- 所有智能体的输出均采用 JSON 结构化格式，保证下游智能体能可靠解析
- 提示词中包含详细的角色描述、任务规范、输出格式约束
- 针对不同问题类型（computation/choice/short_answer/proof）有差异化指令

### 3.4 安全工具调用机制

- Calculator 使用 AST 白名单机制，防止任意代码执行
- Python REPL 使用沙盒化的 `exec`，仅暴露安全的 `math` 模块
- 所有工具调用都有异常捕获和友好的错误返回

# 实验结果及分析

## 1. 实验结果展示

### 1.1 测试用例 1：定积分计算 (computation)

**问题**: Compute the value of $\int_0^1 x^2\,dx$

**求解过程**:

```
[Orchestrator] Classified as: computation
[Orchestrator] Rationale: This is a numerical computation involving 
               definite integration

[Solver/ReAct] Completed 2 ReAct steps
[Solver/ReAct] Used tools: ['sympy_math', 'python_repl']
  - sympy_math: integrate(x**2, (x, 0, 1)) → x**3/3 evaluated: 1/3
  - python_repl: Numerical Riemann sum verification → ~0.333333

[Validator] Score: 50/100
[Validator] Correct: True
```

**最终答案**: $\int_0^1 x^2\,dx = 1/3$ ✅

### 1.2 测试用例 2：无理数证明 (proof)

**问题**: Prove that √2 is irrational

**求解过程**:
```
[Orchestrator] Classified as: proof
[Prover/ToT] Generated 4 proof strategies
[Prover/ToT] Evaluated strategies:
  - Strategy 1 (contradiction): Correctness=9, Completeness=8, Feasibility=10
  - Strategy 2 (prime factorization): Correctness=8, Completeness=7, Feasibility=8
  - Strategy 3 (continued fractions): Correctness=7, Completeness=6, Feasibility=6
  - Strategy 4 (geometric): Correctness=6, Completeness=6, Feasibility=7
[Prover/ToT] Selected: Proof by contradiction
[Prover/ToT] Expanded into full proof (using sympy_math for verification)
  - sympy_math: expand((2*k)**2) → 4*k**2
  - sympy_math: simplify((2*k)**2/2) → 2*k**2

[Validator] Score: 50/100
[Validator] Correct: True
```

**最终答案**: 通过反证法证明 √2 为无理数 ✅

### 1.3 测试用例 3：微分方程选择题 (choice)

**问题**: Which of the following is the general solution to $\frac{dy}{dx} = ky$?
A) $y = kx + C$  B) $y = Ce^(kx)$  C) $y = k^x + C$  D) $y = C/x$

**最终答案**: **Option B: $y = Ce^{kx}$**

- $\frac{dy}{dx} = C \cdot k \cdot e^{kx} = k \cdot (Ce^{kx}) = ky$
- ✅ Correct! This satisfies the differential equation.

## 2. 评测指标展示及分析

### 2.1 智能体分工评价

| 智能体 | 功能定位 | 提示词适配 | 推理方法 | 工具使用 |
|--------|---------|-----------|---------|---------|
| Orchestrator | 问题分类与路由 | 结构化JSON输出规范 | Chain-of-Thought | 无 |
| Prover | 证明题求解 | 数学证明标准格式 | Tree of Thoughts | SymPy, Web Search |
| Solver | 计算/选择/简答 | 类型差异化指令 | ReAct | Python REPL, SymPy, Calculator, Web Search |
| Validator | 解答验证评分 | 多维度评分标准 | Chain-of-Thought | Python REPL |
| Reflector | 失败分析与改进 | 根本原因分析框架 | Reflection | 无 |

所有 5 个智能体均有明确定位，提示词适配各自任务，分工合理。

### 2.2 推理方法分析

**Tree of Thoughts (ToT)**:

- 广度参数: 4 种策略并行生成
- 评估维度: Correctness, Completeness, Feasibility (各 1-10 分)
- 优势: 避免单一证明路径的局部最优，通过多策略比较提高证明质量
- 局限: 需要多次 LLM 调用，时延较高

**ReAct**:

- 最大步数: 8 步
- 循环模式: Thought → Action → Observation
- 优势: 工具与推理的紧密耦合，可以进行数值验证

**Reflection**:

- 分析维度: 根本原因、错误模式、知识盲区、改进方案
- 重试机制: 最多 3 次尝试
- 优势: 使系统具有自我纠正能力

### 2.3 记忆系统分析

| 特性 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| 存储后端 | 内存 (LangGraph State + Buffer) | ChromaDB (磁盘持久化) |
| 检索方式 | 按时间顺序 | 语义相似度 |
| 生命周期 | 单次会话 | 跨会话持久化 |
| 容量限制 | 50 条消息 | 无限制（磁盘容量） |
| 嵌入模型 | 无 | all-MiniLM-L6-v2 |
| 典型用途 | 当前问题上下文 | 历史问题参考 |

### 2.4 系统优势与局限

**优势**:

1. 模块化设计：每个智能体独立定义，易于扩展和替换
2. 推理方法多样：针对不同问题类型采用最适合的推理策略
3. 闭环自纠正：验证-反思-重试机制提高最终答案质量
4. 知识积累：长期记忆使系统随着使用逐步提升

**局限**:

1. 时延：ToT 方法需要多次 LLM 调用，响应时间较长
2. 验证精度：LLM 验证器对复杂证明的评判准确度有限
3. 工具链依赖：部分工具（如 Web Search）依赖外部网络服务

---
# 参考资料

- LangGraph 官方文档: [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/)
- Tree of Thoughts (Yao et al., 2023): [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
- ReAct (Yao et al., 2022): [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
- Reflexion (Shinn et al., 2023): [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)
- AlphaProof / AlphaGeometry (Google DeepMind, 2024): [https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- ChromaDB: [https://www.trychroma.com/](https://www.trychroma.com/)
- LangChain: [https://python.langchain.com/](https://python.langchain.com/)
- DeepSeek API: [https://platform.deepseek.com/api-docs](https://platform.deepseek.com/api-docs)
- SymPy: [https://www.sympy.org/](https://www.sympy.org/)
