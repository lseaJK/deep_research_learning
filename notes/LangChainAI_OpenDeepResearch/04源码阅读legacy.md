
## 早期多代理研究概述
在 `legacy/multi_agent.py` 中，早期多代理（Multi-Agent）实现通过**监督者-研究员架构**构建了自动化研究流程，其核心逻辑、与当前版本的区别及亮点如下：

**一、早期多代理实现的核心逻辑**

该实现通过 `langgraph` 的状态图（`StateGraph`）构建了两层工作流：**监督者（Supervisor）** 和**研究员（Researcher）**，分工协作完成报告生成，具体流程如下：

1. **角色与职责划分**

- **监督者（Supervisor）**： 
  作为“总协调者”，负责统筹全局流程：
- 规划报告结构（定义章节列表 `Sections`）；
- 生成报告引言（`Introduction`）和结论（`Conclusion`）；
- 调用搜索工具（如 Tavily/DuckDuckGo）补充全局信息；
- 接收研究员完成的章节，整合为最终报告。
- **研究员（Researcher）**： 
  作为“执行者”，并行处理具体章节：
- 针对监督者分配的单个章节（如“技术原理”“市场分析”）进行专项研究；
- 调用搜索工具获取该章节所需的具体信息；
- 撰写章节内容（`Section`）并返回给监督者。
  
2. **工作流执行逻辑**
  
通过两个嵌套的状态图实现流程控制：
- **研究员子图（`research_builder`）**： 
  单个研究员的工作闭环：`START → research_agent（决策是否搜索/撰写）→ research_agent_tools（执行工具调用）→ 循环直至章节完成 → END`。 
  每个研究员独立处理一个章节，支持并行执行（通过 `Send` 机制分发任务）。
- **监督者主图（`supervisor_builder`）**： 
  全局流程控制：`START → supervisor（规划章节/生成引言）→ supervisor_tools（执行工具调用）→ 分发章节给 research_team（研究员子图）→ 收集结果 → 生成结论 → 整合为最终报告 → END`。
  
3. **核心技术细节**
  
- **状态管理**：
- 全局状态（`ReportState`）：存储章节列表、已完成章节、最终报告等；
- 章节状态（`SectionState`）：单个研究员的工作上下文（如当前章节名称、搜索结果）。
- **工具集成**：
- 基础工具：搜索（Tavily/DuckDuckGo）、章节生成（`Section`）、报告结构定义（`Sections`）等；
- 扩展工具：通过 MCP（Model Context Protocol）集成外部工具（如数据库查询、API调用）。
- **结构化输出**： 
  用 Pydantic 模型（`Section`/`Introduction`/`Conclusion`）强制规范报告格式，确保章节名称、内容结构一致。
  
**二、与当前版本的区别**
  
根据项目文档和代码对比，早期多代理实现与当前版本的核心区别如下：

| **维度** | **早期多代理实现（`legacy/multi_agent.py`）** | **当前版本** |
| --- | --- | --- |
| **性能** | 性能较低（文档明确提及“less performant”）。 | 性能更优，在 Deep Research Bench 排名第6（0.4344分）。 |
| **并行粒度** | 以“章节”为单位并行（每个研究员处理一个章节）。 | 可能采用更细粒度的并行（如多步骤任务拆分）。 |
| **搜索工具支持** | 仅支持 Tavily/DuckDuckGo/None，其他工具需用 graph 实现。 | 支持更多搜索工具（如 OpenAI/Anthropic 原生搜索）。 |
| **灵活性** | 流程固定（监督者→研究员→整合），人工干预较少。 | 更灵活的动态调整（如动态增减研究步骤、自适应工具选择）。 |
| **代码架构** | 状态图嵌套（监督者包含研究员子图），逻辑相对固定。 | 可能采用更模块化的设计，便于扩展和配置。 |

**三、早期实现的亮点**

尽管性能不及当前版本，早期多代理实现仍有以下设计亮点：

1. **明确的角色分工**： 
监督者与研究员权责分离，监督者专注全局规划，研究员专注专项执行，符合真实研究场景的分工逻辑。
2. **并行效率提升**： 
多个研究员可同时处理不同章节（通过 `Send` 机制分发任务），相比串行处理大幅缩短报告生成时间，适合对速度敏感的场景。
3. **结构化输出保障**： 
通过 Pydantic 模型强制规范章节、引言、结论的格式，确保最终报告结构清晰、风格统一。
4. **MCP 工具扩展**： 
支持通过 MCP 协议集成外部工具（如私有数据库、定制 API），扩展了研究能力，适配复杂场景（如企业内部数据查询）。
5. **可配置性**： 
通过 `MultiAgentConfiguration` 支持灵活调整模型（监督者/研究员模型）、搜索工具、是否启用 MCP 等，适配不同需求。

**总结**

早期多代理实现通过“监督者-研究员”的并行架构，在保证报告结构化的同时提升了生成效率，为后续版本的优化提供了“角色分工”和“并行处理”的设计基础。其核心局限在于性能和工具支持范围，而当前版本在此基础上进行了架构优化和功能扩展。

## 早期多代理实现

`open_deep_research/src/legacy/multi_agent.py` 实现了一个基于“监督者-研究员”架构的多代理自动化研究系统，核心目标是通过分工协作生成结构化研究报告。以下结合代码细节，从**核心架构、数据模型、代理逻辑、工作流控制**四个维度进行详细说明：

### **一、核心架构：监督者与研究员的分工协作**

该实现通过 LangGraph 的 `StateGraph` 构建了两层嵌套工作流，形成“1 个监督者 + N 个研究员”的多代理系统：

- **监督者（Supervisor）**：负责全局规划（如定义报告结构、生成引言/结论）、工具调用协调（如搜索）、任务分发（将章节分配给研究员）。
- **研究员（Researcher）**：作为子代理，并行处理单个章节的研究（如搜索信息、撰写内容），最终将结果返回给监督者。 
  这种架构通过“分工+并行”提升报告生成效率，符合 legacy 版本“速度优化”的设计目标（参考之前提到的 legacy 特性）。
  
### **二、数据模型：结构化报告元素的定义**

代码通过 Pydantic 模型强制规范报告的核心元素，确保数据格式一致，便于在代理间传递和整合：

| 模型类 | 作用  | 关键字段 |
| --- | --- | --- |
| `Section` | 单个章节的内容封装 | `name`（章节名）、`description`（研究范围）、`content`（章节内容） |
| `Sections` | 报告章节列表（用于监督者规划结构） | `sections`（章节名列表，如 `["背景", "现状分析"]`） |
| `Introduction` | 报告引言 | `name`（报告标题）、`content`（引言内容） |
| `Conclusion` | 报告结论 | `name`（结论标题）、`content`（结论内容） |
| `Question` | 监督者向用户发起的澄清问题（用于明确研究范围） | `question`（具体问题） |
| `FinishResearch`/`FinishReport` | 无操作工具，用于标记研究/报告完成状态（流程控制信号） | 无实际字段，仅作为状态标识 |

```python
class Section(BaseModel):
    """Section of the report."""
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Research scope for this section of the report.",
    )
    content: str = Field(
        description="The content of the section."
    )

class Sections(BaseModel):
    """List of section titles of the report."""
    sections: List[str] = Field(
        description="Sections of the report.",
    )

class Introduction(BaseModel):
    """Introduction to the report."""
    name: str = Field(
        description="Name for the report.",
    )
    content: str = Field(
        description="The content of the introduction, giving an overview of the report."
    )

class Conclusion(BaseModel):
    """Conclusion to the report."""
    name: str = Field(
        description="Name for the conclusion of the report.",
    )
    content: str = Field(
        description="The content of the conclusion, summarizing the report."
    )


# No-op tool to indicate that the research is complete
class FinishResearch(BaseModel):
    """Finish the research."""

# No-op tool to indicate that the report writing is complete
class FinishReport(BaseModel):
    """Finish the report."""

## State
class ReportStateOutput(MessagesState):
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class ReportState(MessagesState):
    sections: list[str] # List of report sections 
    completed_sections: Annotated[list[Section], operator.add] # Send() API key
    final_report: str # Final report
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: Annotated[str, operator.add] # String of formatted source content from web search

class SectionState(MessagesState):
    section: str # Report section  
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    # for evaluation purposes only
    # this is included only if configurable.include_source_str is True
    source_str: str # String of formatted source content from web search

```

### **三、工具系统：动态加载与代理专属工具集**

工具是代理与外部交互的核心（如搜索、数据查询），代码通过工具工厂函数实现动态配置，确保不同代理仅获取所需工具：

#### 1. **搜索工具适配**

`get_search_tool` 函数根据配置（`search_api`）选择搜索工具，支持 Tavily、DuckDuckGo 或禁用搜索：

```python
def get_search_tool(config: RunnableConfig):
search_api = get_config_value(configurable.search_api)
if search_api.lower() == "tavily":
search_tool = tavily_search # 绑定Tavily搜索工具
elif search_api.lower() == "duckduckgo":
search_tool = duckduckgo_search # 绑定DuckDuckGo搜索工具
# 其他工具需通过graph.py实现（legacy版本限制）
```

#### 2. **代理专属工具集**

- **监督者工具（`get_supervisor_tools`）**：聚焦全局规划，包括：
- 结构工具：`Sections`（定义章节）、`Introduction`（生成引言）、`Conclusion`（生成结论）；
- 交互工具：`Question`（用户澄清）、`FinishReport`（结束报告）；
- 扩展工具：搜索工具（如Tavily）、MCP工具（通过`_load_mcp_tools`加载外部工具）。
- **研究员工具（`get_research_tools`）**：聚焦单章节处理，包括：
- 内容工具：`Section`（撰写章节）、`FinishResearch`（结束研究）；
- 扩展工具：搜索工具、MCP工具（与监督者共享，但权限更聚焦）。

### **四、代理逻辑：核心函数与工作流程**

#### 1. **监督者（Supervisor）逻辑**

监督者通过 `supervisor` 函数和 `supervisor_tools` 函数实现“规划-分发-整合”闭环：

```python

async def supervisor(state: ReportState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""

    # Messages
    messages = state["messages"]

    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    supervisor_model = get_config_value(configurable.supervisor_model)

    # Initialize the model
    llm = init_chat_model(model=supervisor_model)
    
    # If sections have been completed, but we don't yet have the final report, then we need to initiate writing the introduction and conclusion
    if state.get("completed_sections") and not state.get("final_report"):
        research_complete_message = {"role": "user", "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        messages = messages + [research_complete_message]

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    
    
    llm_with_tools = (
        llm
        .bind_tools(
            supervisor_tool_list,
            parallel_tool_calls=False,
            # force at least one tool call
            tool_choice="any"
        )
    )

    # Get system prompt
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today_str())
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Invoke
    return {
        "messages": [
            await llm_with_tools.ainvoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                + messages
            )
        ]
    }

async def supervisor_tools(state: ReportState, config: RunnableConfig)  -> Command[Literal["supervisor", "research_team", "__end__"]]:
    """Performs the tool call and sends to the research agent"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    sections_list = []
    intro_content = None
    conclusion_content = None
    source_str = ""

    # Get tools based on configuration
    supervisor_tool_list = await get_supervisor_tools(config)
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    search_tool_names = {
        tool.name
        for tool in supervisor_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }

    # First process all tool calls to ensure we respond to each one (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = supervisor_tools_by_name[tool_call["name"]]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # Store special tool results for processing after all tools have been called
        if tool_call["name"] == "Question":
            # Question tool was called - return to supervisor to ask the question
            question_obj = cast(Question, observation)
            result.append({"role": "assistant", "content": question_obj.question})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "FinishReport":
            result.append({"role": "user", "content": "Report is Finish"})
            return Command(goto=END, update={"messages": result})
        elif tool_call["name"] == "Sections":
            sections_list = cast(Sections, observation).sections
        elif tool_call["name"] == "Introduction":
            # Format introduction with proper H1 heading if not already formatted
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name}\n\n{observation.content}"
            else:
                intro_content = observation.content
        elif tool_call["name"] == "Conclusion":
            # Format conclusion with proper H2 heading if not already formatted
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content
        elif tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)

    # After processing all tool calls, decide what to do next
    if sections_list:
        # Send the sections to the research agents
        return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
    elif intro_content:
        # Store introduction while waiting for conclusion
        # Append to messages to guide the LLM to write conclusion next
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})
        state_update = {
            "final_report": intro_content,
            "messages": result,
        }
    elif conclusion_content:
        # Get all sections and combine in proper order: Introduction, Body Sections, Conclusion
        intro = state.get("final_report", "")
        body_sections = "\n\n".join([s.content for s in state["completed_sections"]])
        
        # Assemble final report in correct order
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        
        # Append to messages to indicate completion
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})

        state_update = {
            "final_report": complete_report,
            "messages": result,
        }
    else:
        # Default case (for search tools, etc.)
        state_update = {"messages": result}

    # Include source string for evaluation
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return Command(goto="supervisor", update=state_update)

async def supervisor_should_continue(state: ReportState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]
    # End because the supervisor asked a question or is finished
    if not last_message.tool_calls:
        # Exit the graph
        return END

    # If the LLM makes a tool call, then perform an action
    return "supervisor_tools"

```

- **步骤1：全局规划（`supervisor` 函数）** 
加载模型（如Claude），根据当前状态（如是否已有完成章节）生成工具调用指令：
- 若未开始：调用 `Sections` 工具定义章节结构，或调用 `Question` 工具澄清用户需求；
- 若章节已完成：触发“生成引言和结论”的提示（通过追加用户消息引导模型）。 
关键代码：

```python
# 若章节已完成但无最终报告，引导模型生成引言和结论
if state.get("completed_sections") and not state.get("final_report"):
research_complete_message = {"role": "user", "content": "Research is complete. Now write introduction and conclusion..."}
messages = messages + [research_complete_message]
```

- **步骤2：工具执行与结果处理（`supervisor_tools` 函数）** 
执行监督者的工具调用，并根据结果更新状态或分发任务：
- 若调用 `Sections` 工具：提取章节列表，通过 `Send` 机制分发每个章节给 `research_team`（研究员子图）；
- 若调用 `Introduction`/`Conclusion` 工具：格式化内容（如添加Markdown标题），暂存到状态中；
- 若调用 `Question` 工具：直接向用户返回澄清问题，终止当前流程。 
关键代码（任务分发）：

```python
if sections_list:
# 向研究员子图发送每个章节任务（并行处理）
return Command(goto=[Send("research_team", {"section": s}) for s in sections_list], update={"messages": result})
```

- **步骤3：流程控制（`supervisor_should_continue` 函数）** 
根据最后一条消息是否包含工具调用，决定继续流程或终止：
- 若有工具调用：进入 `supervisor_tools` 节点执行；
- 若无工具调用：流程结束（如已生成完整报告）。

#### 2. **研究员（Researcher）逻辑**

研究员通过 `research_agent` 函数和 `research_agent_tools` 函数实现“搜索-撰写”闭环，每个研究员独立处理一个章节：

```python

async def research_agent(state: SectionState, config: RunnableConfig):
    """LLM decides whether to call a tool or not"""
    
    # Get configuration
    configurable = MultiAgentConfiguration.from_runnable_config(config)
    researcher_model = get_config_value(configurable.researcher_model)
    
    # Initialize the model
    llm = init_chat_model(model=researcher_model)

    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=state["section"],
        number_of_queries=configurable.number_of_queries,
        today=get_today_str(),
    )
    if configurable.mcp_prompt:
        system_prompt += f"\n\n{configurable.mcp_prompt}"

    # Ensure we have at least one user message (required by Anthropic)
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]

    return {
        "messages": [
            # Enforce tool calling to either perform more search or call the Section tool to write the section
            await llm.bind_tools(research_tool_list,             
                                 parallel_tool_calls=False,
                                 # force at least one tool call
                                 tool_choice="any").ainvoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                + messages
            )
        ]
    }

async def research_agent_tools(state: SectionState, config: RunnableConfig):
    """Performs the tool call and route to supervisor or continue the research loop"""
    configurable = MultiAgentConfiguration.from_runnable_config(config)

    result = []
    completed_section = None
    source_str = ""
    
    # Get tools based on configuration
    research_tool_list = await get_research_tools(config)
    research_tools_by_name = {tool.name: tool for tool in research_tool_list}
    search_tool_names = {
        tool.name
        for tool in research_tool_list
        if tool.metadata is not None and tool.metadata.get("type") == "search"
    }
    
    # Process all tool calls first (required for OpenAI)
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = research_tools_by_name[tool_call["name"]]
        # Perform the tool call - use ainvoke for async tools
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)

        # Append to messages 
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        # Store the section observation if a Section tool was called
        if tool_call["name"] == "Section":
            completed_section = cast(Section, observation)

        # Store the source string if a search tool was called
        if tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)
    
    # After processing all tools, decide what to do next
    state_update = {"messages": result}
    if completed_section:
        # Write the completed section to state and return to the supervisor
        state_update["completed_sections"] = [completed_section]
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return state_update

async def research_agent_should_continue(state: SectionState) -> str:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls[0]["name"] == "FinishResearch":
        # Research is done - return to supervisor
        return END
    else:
        return "research_agent_tools"
    
```

- **步骤1：章节研究（`research_agent` 函数）** 
加载模型（如Claude-3-5），根据章节主题生成工具调用：
- 若信息不足：调用搜索工具获取资料；
- 若信息足够：调用 `Section` 工具撰写章节内容。 
关键代码（强制工具调用）：

```python
# 绑定工具并强制至少一次调用（确保要么搜索要么写章节）
llm.bind_tools(research_tool_list, tool_choice="any")
```

- **步骤2：工具执行与结果处理（`research_agent_tools` 函数）** 
执行研究员的工具调用，若生成章节内容（`Section` 工具），则将结果返回给监督者：

```python
if completed_section:
# 将完成的章节写入状态，返回给监督者
state_update["completed_sections"] = [completed_section]
```

- **步骤3：流程控制（`research_agent_should_continue` 函数）** 
根据工具调用类型决定是否继续：
- 若调用 `FinishResearch`：终止当前研究员流程，返回结果；
- 其他工具（如搜索）：继续循环（`research_agent_tools` → `research_agent`）。

### **五、工作流构建：LangGraph 图结构设计**

通过 `StateGraph` 定义节点和边，构建嵌套工作流：
research_builder——research子图
supervisor_builder——总图，调用research_builder

```python
# Research agent workflow
research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=MultiAgentConfiguration)
research_builder.add_node("research_agent", research_agent)
research_builder.add_node("research_agent_tools", research_agent_tools)
research_builder.add_edge(START, "research_agent") 
research_builder.add_conditional_edges(
    "research_agent",
    research_agent_should_continue,
    ["research_agent_tools", END]
)
research_builder.add_edge("research_agent_tools", "research_agent")

# Supervisor workflow
supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=MultiAgentConfiguration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("research_team", research_builder.compile())

# Flow of the supervisor agent
supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_conditional_edges(
    "supervisor",
    supervisor_should_continue,
    ["supervisor_tools", END]
)
supervisor_builder.add_edge("research_team", "supervisor")

graph = supervisor_builder.compile()
```

1. **研究员子图（`research_builder`）** 
单个研究员的工作闭环： 
`START → research_agent（决策）→ research_agent_tools（执行工具）→ 循环直至完成 → END`
2. **监督者主图（`supervisor_builder`）** 
全局流程控制： 
`START → supervisor（规划）→ supervisor_tools（执行工具/分发任务）→ research_team（研究员子图）→ 整合结果 → END`

### **六、核心特点总结**

3. **并行效率**：多个研究员同时处理不同章节（通过 `Send` 机制），缩短报告生成时间；
4. **结构化输出**：通过 Pydantic 模型强制报告格式（如章节标题、引用规范）；
5. **灵活配置**：支持切换搜索工具、MCP外部工具、模型类型（监督者/研究员可使用不同模型）；
6. **流程可控**：通过“工具调用+状态判断”实现动态流程调整（如中途澄清需求、补充搜索）。 
该实现是 legacy 版本中“速度优先”的典型代表，但其工具支持范围（仅限Tavily/DuckDuckGo）和架构灵活性低于当前版本。