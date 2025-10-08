# 1. scope
## 1.1 澄清(用户意图)
根据模型返回结果决定是直接进入研究阶段还是向用户请求更多信息
```python
# src\open_deep_research\deep_researcher.py
async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

# src\open_deep_research\prompts.py
clarify_with_user_instructions="""
These are the messages that have been exchanged so far from the user asking for the report:
<Messages>
{messages}
</Messages>

Today's date is {date}.

Assess whether you need to ask a clarifying question, or if the user has already provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a clarifying question, you almost always do not need to ask another one. Only ask another question if ABSOLUTELY NECESSARY.

If there are acronyms, abbreviations, or unknown terms, ask the user to clarify.
If you need to ask a question, follow these guidelines:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity. Make sure that this uses markdown formatting and will be rendered correctly if the string output is passed to a markdown renderer.
- Don't ask for unnecessary information, or information that the user has already provided. If you can see that the user has already provided the information, do not ask for it again.

Respond in valid JSON format with these exact keys:
"need_clarification": boolean,
"question": "<question to ask the user to clarify the report scope>",
"verification": "<verification message that we will start research>"

If you need to ask a clarifying question, return:
"need_clarification": true,
"question": "<your clarifying question>",
"verification": ""

If you do not need to ask a clarifying question, return:
"need_clarification": false,
"question": "",
"verification": "<acknowledgement message that you will now start research based on the provided information>"

For the verification message when no clarification is needed:
- Acknowledge that you have sufficient information to proceed
- Briefly summarize the key aspects of what you understand from their request
- Confirm that you will now begin the research process
- Keep the message concise and professional
"""
```

## 1.2 将用户的查询转换为结构化的研究简报，并初始化研究主管的上下文
将用户输入转换为明确的研究主题和范围
为研究主管设置系统提示和初始上下文
传递研究简报至下一个阶段

```python
async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )

transform_messages_into_research_topic_prompt = """You will be given a set of messages that have been exchanged so far between yourself and the user. 
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include all known user preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Fill in Unstated But Necessary Dimensions as Open-Ended
- If certain attributes are essential for a meaningful output but the user has not provided them, explicitly state that they are open-ended or default to no specific constraint.

3. Avoid Unwarranted Assumptions
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat it as flexible or accept all possible options.

4. Use the First Person
- Phrase the request from the perspective of the user.

5. Sources
- If specific sources should be prioritized, specify them in the research question.
- For product and travel research, prefer linking directly to official or primary websites (e.g., official brand sites, manufacturer pages, or reputable e-commerce platforms like Amazon for user reviews) rather than aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original paper or official journal publication rather than survey papers or secondary summaries.
- For people, try linking directly to their LinkedIn profile, or their personal website if they have one.
- If the query is in a specific language, prioritize sources published in that language.
"""
```

### 1.3 研究主管负责规划研究策略并将任务委派给研究员
配置主管模型及可用工具（研究委派、完成信号、战略思考）
基于当前上下文生成研究策略
跟踪研究迭代次数
```python
lead_researcher_prompt = """You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool. For context, today's date is {date}.

<Task>
Your focus is to call the "ConductResearch" tool to conduct research against the overall research question passed in by the user. 
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
3. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless the user request has clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to ConductResearch and think_tool if you cannot find the right sources

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I delegate more research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
**Simple fact-finding, lists, and rankings** can use a single sub-agent:
- *Example*: List the top 10 coffee shops in San Francisco → Use 1 sub-agent

**Comparisons presented in the user request** can use a sub-agent for each element of the comparison:
- *Example*: Compare OpenAI vs. Anthropic vs. DeepMind approaches to AI safety → Use 3 sub-agents
- Delegate clear, distinct, non-overlapping subtopics

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather information
- When calling ConductResearch, provide complete standalone instructions - sub-agents can't see other agents' work
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>"""
```
### 1.3.1 supervisor 函数
研究主管负责规划研究策略并将任务委派给研究员。
配置主管模型及可用工具（研究委派、完成信号、战略思考）
基于当前上下文生成研究策略
跟踪研究迭代次数
```python
async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )
```
在 supervisor 函数中，处理完研究策略规划后，会返回一个 Command 对象，明确指定下一步流转到 supervisor_tools 节点

### 1.3.2 supervisor_tools 函数
处理主管调用的工具，包括研究委派和战略思考。
检查研究是否达到终止条件（迭代次数超限、无工具调用、明确完成信号）
处理工具调用结果并更新状态
决定是继续研究循环还是结束研究阶段
```python
async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1：提取状态与检查终止条件：首先从当前状态和配置中提取关键信息，并判断是否满足研究终止条件
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # 定义终止条件
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations  # 迭代次数超限
    no_tool_calls = not most_recent_message.tool_calls  # 无工具调用（异常情况）
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )  # 调用了"研究完成"工具

    
    # 若满足任意终止条件，直接结束研究流程，并整理已有研究笔记
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: 若未终止，则按工具类型（think_tool 和 ConductResearch）分别处理：
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # 2.1 处理 think_tool（战略思考工具）
    # think_tool 用于监督者的战略反思（如调整研究方向、记录思路），无需实际执行外部任务，仅需记录反思内容
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    for tool_call in think_tool_calls: # 为每个反思生成工具反馈消息
        reflection_content = tool_call["args"]["reflection"]  # 提取反思内容
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",  # 记录反思
            name="think_tool",
            tool_call_id=tool_call["id"]  # 关联工具调用ID，确保上下文连贯
        ))
    
    # 2.2 处理 ConductResearch（委派研究任务）
    # ConductResearch 用于将研究任务委派给子研究员（researcher_subgraph），是并行研究的核心逻辑
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    if conduct_research_calls:
        try:
            # 限制并发任务数，超出部分标记为溢出
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # 为每个任务创建子研究员调用任务
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],  # 研究员初始消息
                    "research_topic": tool_call["args"]["research_topic"]  # 具体研究主题
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)  # 并行执行并收集结果

            # 处理研究结果：将子研究员返回的结果封装为 ToolMessage，更新到监督者对话历史
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research"),  # 研究员压缩后的结果
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            
            # 处理溢出任务：对超出并发限制的任务，返回错误提示
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # 收集原始笔记：将所有子研究员的原始研究笔记聚合，用于后续整理
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]  # 更新原始笔记

        # Step 3：错误处理，若执行研究任务时发生错误（如模型令牌限制超限），直接终止研究流程
        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 4：更新状态并决定下一步
    # 将所有工具执行结果（all_tool_messages）更新到监督者对话历史，并返回 supervisor 节点继续研究循环
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor", # 回到监督者节点，继续下一轮规划
        update=update_payload
    ) 

```
supervisor_tools 节点处理完工具调用（如 ConductResearch、think_tool 等）后，同样通过返回 Command 控制后续流转：
若需要继续研究，返回 Command(goto="supervisor")，回到 supervisor 节点进行下一轮规划。
若满足终止条件（如迭代次数超限、收到 ResearchComplete 信号），返回 Command(goto=END)，结束工作流。
这形成了 supervisor → supervisor_tools → supervisor 的循环，直到研究完成。

### 1.3.3 supervisor_builder
Supervisor（研究主管）和 Supervisor_tools（主管工具执行器）构成了研究流程的核心控制循环，二者通过 "规划 - 执行 - 反馈" 的模式实现协同工作，具体交互逻辑如下：
初始启动：工作流从supervisor节点开始，主管根据研究简报（research_brief）制定初步研究策略
**工具调用决策**：
Supervisor 分析当前上下文，决定需要调用的工具（如ConductResearch委派任务、think_tool战略思考、ResearchComplete结束信号）
生成包含工具调用指令的响应，通过Command将流程导向supervisor_tools
**工具执行阶段**：
Supervisor_tools 接收工具调用指令，分类处理不同类型的工具调用：
- think_tool：直接记录反思内容，生成工具反馈消息
- ConductResearch：并行启动多个研究员子图（researcher_subgraph）执行具体研究任务
- ResearchComplete：触发研究终止条件

收集所有工具执行结果，生成工具消息（ToolMessage）
**状态更新与循环控制**：
Supervisor_tools 将工具执行结果整合到状态中，更新supervisor_messages和raw_notes
检查终止条件（迭代次数超限、无工具调用、收到 ResearchComplete 信号）：
若满足终止条件：流程结束（goto=END），整理研究笔记
若不满足：将流程导回supervisor节点，继续下一轮规划
循环迭代：上述过程反复进行，直到满足终止条件，形成完整的研究闭环


`supervisor_builder` 是基于 LangGraph 的 `StateGraph` 构建的**主管工作流子图**，负责协调深度研究过程中的任务规划、委派与结果整合。它是整个深度研究代理（Deep Research agent）的核心控制组件，通过状态管理和节点流转实现研究过程的自动化与智能化。
```python
# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
# 添加主管节点
supervisor_builder.add_node("supervisor", supervisor) # 主管理逻辑节点
supervisor_builder.add_node("supervisor_tools", supervisor_tools) # 工具执行节点
# 定义工作流入口
supervisor_builder.add_edge(START, "supervisor")
# 编译为可执行子图
supervisor_subgraph = supervisor_builder.compile()
```
其核心作用是：构建一个**状态机工作流**，使主管（Supervisor）能够完成“规划研究策略→委派任务→处理结果→调整策略”的闭环，最终生成满足用户需求的研究成果。

1. **状态定义（`SupervisorState`）**

`supervisor_builder` 以 `SupervisorState` 作为状态载体（来自 `state.py`），包含研究过程中需要跟踪的核心信息：

```python
# src\open_deep_research\state.py
class SupervisorState(TypedDict):
 supervisor_messages: list[MessageLikeRepresentation] # 主管与工具/研究员的对话历史
 research_brief: str # 结构化的研究简报（来自用户查询转换）
 notes: list[str] # 整理后的研究笔记
 research_iterations: int # 研究迭代次数（防止无限循环）
 raw_notes: list[str] # 原始研究结果（未整理）
```

状态在节点间流转时被更新，确保主管始终基于最新上下文决策。

2. **节点（Nodes）**

`supervisor_builder` 包含两个核心节点，分别对应研究过程的不同阶段：
| 节点名称 | 绑定函数 | 功能描述 |
|-------------------|-------------------|--------------------------------------------------------------------------|
| `"supervisor"` | `supervisor` 函数 | 主管的核心逻辑节点：基于研究简报和历史上下文，制定研究策略，决定调用哪些工具（如委派研究任务、战略思考、终止研究）。 |
| `"supervisor_tools"` | `supervisor_tools` 函数 | 工具执行节点：处理 `supervisor` 节点发起的工具调用，执行具体任务（如并行委派研究、记录反思），并判断是否需要继续研究循环。 |

3. **边（Edges）：工作流流转规则**

`supervisor_builder` 通过“边”定义节点间的流转关系，目前核心规则为：

- `supervisor_builder.add_edge(START, "supervisor")`：工作流从 `START` 开始，直接进入 `supervisor` 节点（即研究从主管规划开始）。
- 后续流转由节点返回的 `Command` 决定：
  - `supervisor` 节点执行后，通过 `Command(goto="supervisor_tools")` 流转到 `supervisor_tools` 节点处理工具调用。
  - `supervisor_tools` 节点执行后，根据终止条件判断：若需继续研究，通过 `Command(goto="supervisor")` 回到 `supervisor` 节点；若满足终止条件，通过 `Command(goto=END)` 结束工作流。
    
4. **配置依赖（`Configuration`）**
    
`supervisor_builder` 以 `Configuration` 作为配置 schema（来自 `configuration.py`），通过配置参数控制研究行为，关键参数包括：
- `max_concurrent_research_units`：最大并行研究任务数（控制资源消耗）。
- `max_researcher_iterations`：最大研究迭代次数（防止无限循环）。
- `research_model`：主管使用的大模型（如 `openai:gpt-4.1`）。
- `search_api`：搜索工具类型（如 `tavily`、`openai`）。
  
**工作流运行逻辑**

1. **初始化**：从 `START` 进入 `supervisor` 节点，主管基于 `research_brief`（研究简报）和初始上下文（`supervisor_messages`）开始规划。
2. **策略制定**：`supervisor` 节点调用工具（如 `ConductResearch` 委派任务、`think_tool` 战略思考），生成工具调用指令，流转到 `supervisor_tools` 节点。
3. **工具执行**：`supervisor_tools` 节点执行工具调用：
  - 对 `think_tool`：记录反思内容，生成工具反馈。
  - 对 `ConductResearch`：并行启动多个研究员子图（`researcher_subgraph`）执行具体研究，收集结果。
4. **循环/终止判断**：`supervisor_tools` 检查终止条件（如迭代次数超限、收到 `ResearchComplete` 信号）：
  - 未满足终止条件：将工具结果返回给 `supervisor` 节点，继续下一轮规划。
  - 满足终止条件：结束工作流，整理研究笔记（`notes`）和简报（`research_brief`）。

# 2. research(基于 `researcher` 子图)
researcher（子研究员）是负责执行具体研究任务的核心角色，由 supervisor（监督者）委派特定主题的研究工作。它通过工具调用（如搜索、MCP 工具）收集信息，经过多轮迭代后压缩结果，最终反馈给监督者。

`researcher` 的工作流由 **3 个节点** 和 **动态流转逻辑** 组成，通过 `researcher_builder` 构建为子图（`researcher_subgraph`），可被监督者并行调用。

## 2.1 researcher节点：研究规划与工具调用决策**
该节点是子研究员的“大脑”，负责分析研究主题、规划工具调用策略，生成下一步行动指令。 
```python
async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command( # 返回 `Command(goto="researcher_tools")`，触发工具调用。
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# src\open_deep_research\utils.py
async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Add configured search tools
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    search_tools = await get_search_tool(search_api)
    tools.extend(search_tools)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Add MCP tools if configured
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools


research_system_prompt = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

<Task>
Your job is to use tools to gather information about the user's input topic.
You can use any of the tools provided to you to find resources that can help answer the research question. You can call these tools in series or in parallel, your research is conducted in a tool-calling loop.
</Task>

<Available Tools>
You have access to two main tools:
1. **tavily_search**: For conducting web searches to gather information
2. **think_tool**: For reflection and strategic planning during research
{mcp_prompt}

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps. Do not call think_tool with the tavily_search or any other tools. It should be to reflect on the results of the search.**
</Available Tools>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Start with broader searches** - Use broad, comprehensive queries first
3. **After each search, pause and assess** - Do I have enough to answer? What's still missing?
4. **Execute narrower searches as you gather information** - Fill in the gaps
5. **Stop when you can answer confidently** - Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
"""


```


  
## 2.2 researcher_tools 节点：工具执行与结果处理**
  
该节点负责执行 `researcher` 节点规划的工具调用（如搜索、反思），并根据结果决定继续研究还是终止。 

```python
# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # 并行执行工具：对所有工具调用（如多个搜索查询）进行并行执行（通过 `asyncio.gather`），并封装结果为 `ToolMessage`（关联工具调用 ID，确保上下文连贯）
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)  # 通过 `execute_tool_safely` 捕获工具执行错误（如 API 故障），并返回错误信息供后续处理。
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: 检查终止条件
    # 若未调用任何工具（或未使用原生搜索）、工具调用迭代次数超限（`tool_call_iterations >= max_react_tool_calls`），或主动调用 `ResearchComplete`，则终止研究并进入结果压缩阶段。
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # 若未满足终止条件，返回 `Command(goto="researcher")`，让 `researcher` 节点基于工具结果规划下一轮行动。
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )
```

## 2.3 compress_research节点：研究结果压缩与汇总
  
该节点将多轮工具调用的零散结果（如搜索结果、反思笔记）压缩为简洁、结构化的摘要，同时保留原始笔记。 
- **任务委派**：`supervisor` 通过 `ConductResearch` 工具调用，将研究主题分配给 `researcher_subgraph`，并通过 `asyncio.gather` 实现多 `researcher` 并行执行（受 `max_concurrent_research_units` 限制）。
- **结果反馈**：`researcher` 的压缩结果（`compressed_research`）通过 `ToolMessage` 反馈给 `supervisor`，作为下一轮研究规划的依据。

**执行步骤：**
- **配置压缩模型**：使用 `compression_model`（如 `gpt-4.1`），设置令牌限制（`compression_model_max_tokens`）。
- **准备压缩输入**：收集所有研究过程中的消息（工具输出、AI 反思），添加压缩指令（`compress_research_simple_human_message`）。
- **迭代压缩与容错**：若因令牌超限失败，自动删减早期消息并重试（最多 3 次），确保最终生成摘要。
**输出结果**：返回 `compressed_research`（压缩后的摘要）和 `raw_notes`（原始研究记录），供监督者汇总到最终报告。
```python
async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages) # 删减消息
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }


compress_research_system_prompt = """You are a research assistant that has conducted research on a topic by calling several tools and web searches. Your job is now to clean up the findings, but preserve all of the relevant statements and information that the researcher has gathered. For context, today's date is {date}.

<Task>
You need to clean up information gathered from tool calls and web searches in the existing messages.
All relevant information should be repeated and rewritten verbatim, but in a cleaner format.
The purpose of this step is just to remove any obviously irrelevant or duplicative information.
For example, if three sources all say "X", you could say "These three sources all stated X".
Only these fully comprehensive cleaned findings are going to be returned to the user, so it's crucial that you don't lose any information from the raw messages.
</Task>

<Guidelines>
1. Your output findings should be fully comprehensive and include ALL of the information and sources that the researcher has gathered from tool calls and web searches. It is expected that you repeat key information verbatim.
2. This report can be as long as necessary to return ALL of the information that the researcher has gathered.
3. In your report, you should return inline citations for each source that the researcher found.
4. You should include a "Sources" section at the end of the report that lists all of the sources the researcher found with corresponding citations, cited against statements in the report.
5. Make sure to include ALL of the sources that the researcher gathered in the report, and how they were used to answer the question!
6. It's really important not to lose any sources. A later LLM will be used to merge this report with others, so having all of the sources is critical.
</Guidelines>

<Output Format>
The report should be structured like this:
**List of Queries and Tool Calls Made**
**Fully Comprehensive Findings**
**List of All Relevant Sources (with citations in the report)**
</Output Format>

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
</Citation Rules>

Critical Reminder: It is extremely important that any information that is even remotely relevant to the user's research topic is preserved verbatim (e.g. don't rewrite it, don't summarize it, don't paraphrase it).
"""

```

## 2.4 # Researcher Subgraph Construction
Researcher Subgraph Construction（研究员子图构建）是通过 langgraph 的状态图（StateGraph）实现的，用于定义单个研究员（researcher）的完整工作流程。该子图是多代理架构中执行具体研究任务的核心组件，可被监督者（supervisor）并行调用。


```python
# 子图使用 ResearcherState 作为内部状态，ResearcherOutputState 作为输出状态，并绑定配置类 Configuration 以获取运行时参数（如模型设置、工具配置等）。
class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
# 边定义了节点之间的流转逻辑，决定工作流的执行顺序：
# 从 START 入口直接进入 researcher 节点，启动研究流程。
# 研究结束后（通过 compress_research 节点完成结果压缩），流程终止于 END。
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()
```

# 3. report

final report generation（最终报告生成阶段）是将所有研究阶段收集的信息整合为结构化、全面报告的核心环节。该阶段通过 final_report_generation 函数实现，主要负责将研究员（researcher）和监督者（supervisor）积累的研究结果（如压缩后的摘要、原始笔记）合成为最终报告，并处理潜在的令牌限制等问题。


```python
async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: 提取研究结果与状态清理
    # 收集研究发现：从当前状态（state）中提取所有研究阶段积累的笔记（notes），这些笔记包含了监督者和子研究员的所有研究成果（如压缩后的摘要、工具调用记录等）。
    notes = state.get("notes", [])  # 获取所有研究阶段的笔记
    cleared_state = {"notes": {"type": "override", "value": []}} # 准备清空状态中的 notes 字段（通过 cleared_state），避免后续流程重复使用旧数据。
    findings = "\n".join(notes)     # 将零散笔记拼接为完整文本
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: 生成报告与重试机制（处理令牌限制）
    # 由于研究结果可能包含大量信息（导致令牌超限），该阶段通过 多轮重试 确保报告生成成功。重试逻辑：最多尝试 max_retries（默认 3 次），每次失败后调整输入长度以适应令牌限制。
    max_retries = 3
    current_retry = 0
    findings_token_limit = None # 用于动态调整输入长度的令牌限制
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # 生成成功后，返回包含最终报告内容和清理后状态的字典
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # 令牌超限处理：若因输入内容过长导致令牌超限（is_token_limit_exceeded 检测），则通过 findings_token_limit 减少输入 findings 的长度（例如截断早期内容），重新尝试生成。
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    
# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()
```