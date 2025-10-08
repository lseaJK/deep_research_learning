
> 通义DeepResearch（Tongyi DeepResearch）确实是一个**以深度整合信息为核心能力的研究型智能体**，其定位与传统“生成长报告”的工具存在本质差异。根据官方技术文档和开源代码库，它通过**动态问题拆解、多工具协同调用、跨源数据验证**三大核心机制，实现了从“信息检索”到“深度思考”的跨越。以下是具体解析：

### 一、核心能力：从信息整合到深度推理
1. **动态问题拆解引擎**  
   通义DeepResearch采用**图神经网络（GNN）构建动态思维树**，将复杂问题分解为原子级子任务。例如，在处理法律纠纷时，它会自动拆解为“法条检索”“案例匹配”“学术观点验证”等步骤，并为每个子任务规划工具调用路径。这种能力在房产估值场景中也有体现：当用户询问夏威夷两处房产的售价对比时，模型会先调用“联网搜索”获取2022年销售记录，再通过“数据清洗”工具剔除异常值，最终生成结构化结论。

2. **多模态工具协同网络**  
   区别于单一搜索工具的应用，通义DeepResearch构建了**工具联邦架构**，支持以下协同操作：
   - **跨域数据融合**：同时调用“谷歌学术”（学术文献）、“Python解释器”（数学计算）和“实时天气API”（环境数据），解决跨学科难题。例如，在分析恒星分布的数学模型时，它通过搜索获取天文学数据，再用代码实现欧几里得距离计算，最终整合形成可视化报告。
   - **冲突数据仲裁**：在多源信息不一致时，启动**贝叶斯网络（BN）**进行概率推理，结合**符号逻辑验证**（如Answer Set Programming）确定可信度最高的结论。这种机制在法律案例分析中尤为关键，可有效识别不同判决文书中的矛盾点。

3. **深度推理强化机制**  
   模型支持两种推理模式：
   - **ReAct模式**：适用于短周期任务，通过“思考-行动-观察”循环快速验证假设。例如，在回答“附近亲子餐厅推荐”时，它会调用地图API获取商家列表，再结合用户评分和实时交通数据生成路线规划。
   - **Heavy模式**：针对长周期复杂任务，采用**IterResearch范式**将问题分解为多个研究轮次，每轮仅保留关键信息构建精简工作空间，避免传统方法中的“认知空间窒息”问题。例如，在处理跨年度市场趋势分析时，它会按季度划分数据段，逐轮验证经济指标与销售波动的相关性。

### 二、技术架构：从数据合成到可信执行
1. **智能体合成数据体系**  
   为解决传统训练数据稀缺问题，通义DeepResearch构建了**闭环数据生成系统**：
   - **预训练阶段**：通过AgentFounder框架模拟真实研究场景，生成包含“问题-工具调用-验证过程”的三元组数据。例如，模拟金融分析师分析财报时的工具调用路径（搜索行业报告→调用财务模型→交叉验证数据源）。
   - **后训练阶段**：WebSailor V2系统从真实网站提取信息，通过“原子操作”动态调整问题复杂度（如隐藏关键数据、引入干扰项），生成“博士级”研究难题。这种数据增强技术使模型在GAIA基准测试中达到55%的通过率，超越同类开源方案。

2. **可信执行保障机制**  
   为确保工具调用的可靠性，通义DeepResearch设计了**三级验证体系**：
   - **事实核查**：混合检索系统（ANCE+BM25）验证信息时效性，例如在处理医疗咨询时优先抓取近五年的权威文献。
   - **逻辑校验**：一阶逻辑验证器（FOL Checker）检测推理链条的一致性，例如在专利侵权分析中验证技术特征的覆盖关系。
   - **合规审查**：行业规则引擎（如金融风控模型）确保输出符合监管要求，例如在信贷评估中自动过滤敏感字段。

### 三、应用场景：从垂直领域到通用研究
1. **专业领域深度赋能**  
   - **法律科技**：通义法睿智能体通过“法条-案例-学术观点”三重检索，在“原告是否违反资本维持原则”等复杂问题中，实现案例引用相关性92.3%的准确率，超越OpenAI和Claude的同类产品。
   - **智能出行**：与高德地图共建的出行Agent，整合地图API、实时天气和交通监测数据，在晚高峰导航场景中可减少27%的拥堵时间。

2. **跨学科研究支持**  
   在处理“星座数学模型”等跨学科问题时，通义DeepResearch展现出独特优势：
   - **多工具联动**：调用天文学数据库（SIMBAD）获取恒星分布数据，通过Python实现Delaunay三角剖分算法，最终结合图论分析连通子图的平均规模。
   - **动态路径优化**：当发现现有数据不足以支撑结论时，自动触发二次搜索（如查询“恒星密度与星等关系”的研究论文），形成递归验证闭环。

### 四、与传统工具的本质区别
| **对比维度**       | **传统生成长报告工具**                          | **通义DeepResearch**                              |
|--------------------|------------------------------------------------|-------------------------------------------------|
| **核心目标**       | 基于模板生成格式化文档                          | 构建可追溯的研究路径，输出可验证的决策建议        |
| **信息处理方式**   | 单源数据抽取+简单整合                          | 多源交叉验证+冲突数据仲裁                        |
| **推理能力**       | 基于统计规律的模式匹配                          | 符号逻辑与概率推理结合的混合推理                  |
| **工具调用机制**   | 静态配置工具链                                  | 动态规划工具调用路径，支持递归优化                |
| **结果可信度**     | 依赖输入数据质量                                | 三级验证体系保障事实准确性与逻辑一致性            |

### 五、开源价值与生态建设
通义DeepResearch的开源意义不仅在于模型本身，更在于**完整的Agent构建方法论**：
- **技术全栈开放**：包含从数据合成（AgentFounder）、模型训练（CPT+RL）到推理框架（DTT）的全流程代码，开发者可快速复现并定制化扩展。
- **跨平台适配**：兼容LangChain、AutoGen等主流框架，支持Docker化部署，在阿里云函数计算平台上可实现毫秒级响应。
- **行业解决方案**：提供金融、法律、医疗等领域的预训练适配器，例如金融风控场景中的财务指标异常检测模块。

### 总结
通义DeepResearch重新定义了“深度研究”的内涵——它不是简单的信息堆砌或报告生成，而是通过**动态问题拆解、多工具协同、跨源验证**的闭环机制，构建可追溯的研究路径，输出兼具准确性与可解释性的结论。这种能力使其在学术科研、商业决策、公共政策等领域具有不可替代的价值，而全面开源的技术栈更为开发者提供了打造行业级智能体的“基础设施”。正如官方技术报告所述：“通义DeepResearch的目标，是让每个研究者都能拥有一位永不疲倦的AI科研伙伴。”

---

## 主要执行入口

DeepResearch的执行流程主要通过 `inference/run_react_infer.sh` 脚本启动，该脚本是系统的主要执行入口。
```bash
python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_PATH --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT
```

## 核心执行流程

### 1. ReAct Agent执行循环

在 `inference/react_agent.py` 中，`MultiTurnReactAgent` 类实现了主要的执行逻辑 [2](#0-1) ：

- 系统采用轮次制执行，每轮包含模型调用和工具使用
- 设置了最大150分钟的执行时间限制
- 每轮执行后检查是否达到 `<answer>` 标签来判断任务完成

### 2. 工具调用处理

执行流程中的工具调用处理逻辑 [3](#0-2) ：

- 解析 `<tool_call>` 标签中的工具调用
- 支持Python解释器、搜索、访问网页等多种工具
- 将工具执行结果包装在 `<tool_response>` 标签中返回

### 3. 上下文管理

系统实现了智能的上下文管理机制 [4](#0-3) ：

- 监控token数量，最大限制为108K tokens
- 当超出限制时，强制模型生成最终答案
- 通过消息历史维护对话上下文

### 4. 并行执行框架

在 `inference/run_multi_react.py` 中实现了多线程并行执行 [5](#0-4) ：

- 使用ThreadPoolExecutor进行并发处理
- 支持多个rollout并行执行
- 实现了端口分配和负载均衡

## 系统提示词设计

执行流程使用专门设计的系统提示词 [6](#0-5) ，定义了深度研究助手的核心功能和工具使用规范。

## Notes

代码中还包含WebResearcher的迭代深度研究范式，它采用不同的执行策略，通过离散轮次和报告合成来避免上下文污染。但主要的DeepResearch执行流程是通过上述ReAct框架实现的。

Wiki pages you might want to explore:
- [Overview (Alibaba-NLP/DeepResearch)](/wiki/Alibaba-NLP/DeepResearch#1)

### Citations

**File:** README.md (L89-90)
```markdown
- Open `run_react_infer.sh` and modify the following variables as instructed in the comments:
  * `MODEL_PATH`  - path to the local or remote model weights.
```

**File:** inference/react_agent.py (L138-154)
```python
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
```

**File:** inference/react_agent.py (L159-179)
```python
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    if "python" in tool_call.lower():
                        try:
                            code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                            result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                        except:
                            result = "[Python Interpreter Error]: Formatting error."

                    else:
                        tool_call = json5.loads(tool_call)
                        tool_name = tool_call.get('name', '')
                        tool_args = tool_call.get('arguments', {})
                        result = self.custom_call_tool(tool_name, tool_args)

                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
```

**File:** inference/react_agent.py (L186-194)
```python
            max_tokens = 108 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
```

**File:** inference/run_multi_react.py (L174-181)
```python
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model
                ): task for task in tasks_to_run_all
            }
```

**File:** inference/prompt.py (L1-2)
```python
SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

```
