# PR-Agent 多Agent系统架构分析

## 1. Agent角色分工与职责边界

PR-Agent采用了一种模块化的多Agent协作架构，通过明确的职责分工和边界定义，实现了高效的代码审查与PR管理功能。基于对源码的深入分析，PR-Agent的Agent角色分工主要体现在以下几个方面：

### 1.1 中央调度器：PRAgent类

在PR-Agent的架构中，`PRAgent`类（位于`pr_agent/agent/pr_agent.py`）充当了中央调度器的角色，负责接收用户请求、解析命令、选择合适的专业Agent并协调执行流程。其核心职责包括：

- 请求解析与参数验证：将用户输入的字符串命令解析为结构化的操作指令和参数
- Agent选择与初始化：根据命令类型从`command2class`映射表中选择对应的专业Agent
- 上下文管理：应用仓库特定设置、用户特定设置和语言偏好
- 执行协调：调用选定Agent的`run()`方法并管理执行流程

从实现上看，`PRAgent`采用了命令模式（Command Pattern）的变体，通过`command2class`字典将命令字符串映射到具体的Agent类：

```python
command2class = {
    "review": PRReviewer,
    "describe": PRDescription,
    "improve": PRCodeSuggestions,
    "ask": PRQuestions,
    # 更多命令映射...
}
```

这种设计使得系统可以轻松扩展新的命令和Agent，同时保持接口的一致性。

### 1.2 专业Agent：功能特化的工具类

PR-Agent的核心功能由一系列专业Agent实现，每个Agent负责特定的PR相关任务。这些Agent被实现为独立的类，位于`pr_agent/tools/`目录下：

- **PRReviewer**：负责代码审查，分析PR的质量、安全性和可维护性
- **PRDescription**：生成或优化PR描述
- **PRCodeSuggestions**：提供代码改进建议
- **PRQuestions**：回答关于PR的问题
- **PR_LineQuestions**：处理针对特定代码行的问题
- **PRUpdateChangelog**：更新变更日志
- **PRSimilarIssue**：查找相似的问题
- **PRGenerateLabels**：为PR生成标签
- **PRHelpDocs**：提供文档帮助

每个专业Agent都遵循相似的结构模式，包含初始化方法、运行方法和辅助方法。以`PRReviewer`为例，其主要职责边界包括：

- 获取PR信息（标题、描述、分支、语言等）
- 处理差异文件和补丁
- 构建提示（prompt）并调用AI模型
- 解析和格式化AI响应
- 发布评论或更新PR

这种职责明确的分工使得每个Agent可以专注于自己的核心功能，同时通过统一的接口与系统其他部分交互。

### 1.3 AI处理器：抽象与实现分离

PR-Agent采用了抽象工厂模式来处理与AI模型的交互，通过`BaseAiHandler`抽象类定义统一接口，并由具体实现类如`LiteLLMAIHandler`、`OpenAIAIHandler`等提供实际功能。这种设计实现了AI模型与业务逻辑的解耦，使系统可以灵活切换不同的AI提供商。

`BaseAiHandler`定义了最小化的接口：

```python
class BaseAiHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def deployment_id(self):
        pass
    
    @abstractmethod
    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        pass
```

而具体实现类如`LiteLLMAIHandler`则负责处理与特定AI服务的通信细节、错误重试、响应解析等。这种抽象使得PR-Agent可以无缝支持OpenAI、Claude、Deepseek等多种模型。

## 2. 通信协议与数据格式设计

PR-Agent的Agent间通信采用了一种混合架构，结合了共享状态、参数传递和事件通知等机制。通过分析源码，可以归纳出以下几种主要的通信模式：

### 2.1 基于对象传递的直接通信

在PR-Agent中，Agent之间最主要的通信方式是通过对象引用和方法调用进行直接通信。中央调度器`PRAgent`实例化专业Agent并调用其`run()`方法，同时传递必要的上下文信息：

```python
await command2class[action](pr_url, ai_handler=self.ai_handler, args=args).run()
```

这种通信方式简单直接，但也意味着调用方需要了解被调用方的接口细节，耦合度相对较高。

### 2.2 基于配置的间接通信

PR-Agent大量使用配置系统作为Agent间的间接通信渠道。通过`get_settings()`和`apply_repo_settings()`等函数，不同Agent可以访问和修改共享的配置状态：

```python
# 应用仓库特定设置
apply_repo_settings(pr_url)
# 从参数更新设置
args = update_settings_from_args(args)
# 获取设置并应用
response_language = get_settings().config.get('response_language', 'en-us')
```

这种基于配置的通信模式降低了Agent间的直接依赖，但也引入了全局状态管理的复杂性。

### 2.3 数据格式设计

PR-Agent在Agent间通信中使用了多种数据格式，主要包括：

1. **结构化对象**：如PR信息、文件差异等，通过类实例传递
2. **字典/JSON**：用于灵活表示各种配置和中间结果
3. **字符串模板**：特别是在构建提示（prompt）时使用
4. **标准化响应格式**：AI模型的响应通常被解析为特定的JSON结构

以PRReviewer为例，其构建的变量字典展示了系统内部数据交换的典型格式：

```python
self.vars = {
    "title": self.git_provider.pr.title,
    "branch": self.git_provider.get_pr_branch(),
    "description": self.pr_description,
    "language": self.main_language,
    "diff": "",  # empty diff for initial calculation
    "num_pr_files": self.git_provider.get_num_of_files(),
    # 更多字段...
}
```

这种基于字典的灵活数据结构使得系统可以轻松扩展和修改传递的信息，而不需要频繁更改接口定义。

## 3. 并发控制与任务调度策略

PR-Agent采用了异步编程模型来处理并发任务，主要基于Python的`async/await`机制。通过分析源码，可以识别出以下几种并发控制和任务调度策略：

### 3.1 异步执行模型

PR-Agent的核心方法大多被设计为异步函数，如`PRAgent.handle_request()`和各专业Agent的`run()`方法：

```python
async def handle_request(self, pr_url, request, notify=None) -> bool:
    # 处理逻辑...
    
async def run(self):
    # 执行逻辑...
```

这种设计使得系统可以在等待I/O操作（如API调用、文件读写）时释放控制权，提高整体吞吐量。

### 3.2 任务优先级管理

虽然PR-Agent没有实现显式的任务优先级队列，但通过配置系统和命令处理逻辑，实现了一种隐式的优先级管理。例如，在GitHub App配置中，可以看到命令执行的优先顺序：

```python
pr_commands = [
    "/describe --pr_description.final_update_message=false",
    "/review",
    "/improve",
]
```

这种配置驱动的执行顺序确保了关键任务（如描述生成）先于其他任务执行。

### 3.3 并发限制与资源管理

PR-Agent通过多种机制控制并发度和资源使用：

1. **令牌限制**：使用`TokenHandler`类管理AI模型的令牌使用，防止超出限制
2. **重试策略**：对可能失败的操作（如API调用）实现指数退避重试
3. **超时控制**：为长时间运行的操作设置超时限制

特别是在与AI模型交互时，系统实现了复杂的重试逻辑：

```python
@retry(
    retry=(
        retry_if_exception_type(openai.error.Timeout) |
        retry_if_exception_type(openai.error.APIError) |
        retry_if_exception_type(openai.error.APIConnectionError) |
        retry_if_exception_type(openai.error.RateLimitError) |
        retry_if_exception_type(requests.exceptions.RequestException)
    ),
    stop=stop_after_attempt(OPENAI_RETRIES),
)
async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
    # 实现逻辑...
```

这种细粒度的错误处理和重试机制提高了系统的稳定性和可靠性。

## 4. 容错机制与状态恢复方案

PR-Agent实现了多层次的容错机制，以应对各种可能的故障场景。通过分析源码，可以识别出以下几种主要的容错策略：

### 4.1 异常处理与日志记录

系统广泛使用了异常处理机制，结合详细的日志记录，以捕获和处理各种错误情况：

```python
try:
    # 操作逻辑
except Exception as e:
    get_logger().error(f"Error occurred: {str(e)}")
    # 错误处理逻辑
```

特别是在与外部系统（如Git提供商API、AI模型API）交互时，这种模式被广泛应用。

### 4.2 降级策略与备选方案

PR-Agent实现了多种降级策略，当首选方法失败时自动切换到备选方案：

1. **模型降级**：当高级模型失败时尝试使用功能较简单的模型
2. **功能降级**：当完整功能不可用时提供简化版本
3. **API降级**：当主要API端点不可用时尝试备用端点

例如，在处理内联评论时，系统会在主要方法失败后尝试备选方法：

```python
if get_settings().github.publish_inline_comments_fallback_with_verification:
    # 尝试备选方法发布内联评论
```

### 4.3 幂等操作设计

PR-Agent的许多操作被设计为幂等的，这意味着即使操作被重复执行，也不会产生意外的副作用。这种设计对于处理重试和恢复场景特别重要。

例如，在发布评论前，系统会检查是否已存在相同内容的评论，避免重复：

```python
# 检查是否已存在相同评论
existing_comments = self.git_provider.get_issue_comments()
for comment in existing_comments:
    if comment.body == body:
        return  # 避免重复评论
```

### 4.4 状态持久化与恢复

虽然PR-Agent主要是无状态设计，但对于某些长时间运行的任务，系统实现了简单的状态持久化机制，通过文件系统或数据库存储中间状态，以便在故障后恢复：

```python
# 保存中间状态
with open(cache_file, 'w') as f:
    json.dump(state, f)

# 恢复状态
if os.path.exists(cache_file):
    with open(cache_file, 'r') as f:
        state = json.load(f)
```

这种机制在处理大型PR或需要多步骤处理的任务时特别有用。

## 5. Prompt管理实现机制

PR-Agent的一个核心特性是其复杂而灵活的prompt管理系统，这使得系统能够有效地引导AI模型生成高质量、结构化的输出。通过分析源码，可以识别出以下几种prompt管理策略：

### 5.1 模板化Prompt设计

PR-Agent使用模板化的方法构建prompt，将固定的指令结构与动态的上下文信息结合：

```python
prompt = f"""
You are an experienced code reviewer reviewing a pull request.
Title: {self.vars["title"]}
Branch: {self.vars["branch"]}
Description: {self.vars["description"]}
Language: {self.vars["language"]}
Diff:
{self.vars["diff"]}
...
"""
```

这种模板化设计使得prompt结构清晰可维护，同时允许动态插入特定PR的信息。

### 5.2 分层Prompt组织

PR-Agent的prompt通常被组织为多个层次，包括：

1. **角色定义**：明确AI应扮演的角色（如代码审查员）
2. **上下文信息**：提供PR的相关背景
3. **任务指令**：明确要求AI执行的具体任务
4. **输出格式指导**：规定期望的输出结构
5. **额外约束**：添加特定的限制或偏好

这种分层组织使得prompt既全面又结构化，有助于引导AI生成符合预期的输出。

### 5.3 配置驱动的Prompt定制

PR-Agent允许通过配置文件定制prompt的各个方面，使系统可以适应不同的使用场景和偏好：

```python
# 从配置中获取额外指令
extra_instructions = get_settings().pr_reviewer.extra_instructions

# 将额外指令添加到prompt
if extra_instructions:
    prompt += f"\n\nAdditional instructions: {extra_instructions}"
```

这种配置驱动的方法使得用户可以在不修改代码的情况下调整系统行为。

### 5.4 JSON Schema约束

为了确保AI输出的结构化和一致性，PR-Agent广泛使用JSON Schema来约束响应格式：

```python
# 定义期望的输出格式
output_format = {
    "review_comment": "Detailed review with findings and suggestions",
    "score": "Score from 0-100",
    "findings": [
        {
            "title": "Finding title",
            "description": "Detailed description",
            "severity": "One of: 'High', 'Medium', 'Low', 'Info'"
        }
    ]
}

# 在prompt中包含输出格式要求
prompt += f"\n\nPlease provide your response in the following JSON format:\n{json.dumps(output_format, indent=2)}"
```

这种方法使得系统可以可靠地解析和处理AI的响应，而不需要复杂的后处理逻辑。

## 6. Agent调度器伪代码解读

PR-Agent的核心调度逻辑位于`PRAgent.handle_request()`方法中，下面是该方法的伪代码级解读：

```python
async def handle_request(pr_url, request, notify=None):
    # 步骤1: 应用仓库特定设置
    apply_repo_settings(pr_url)
    
    # 步骤2: 解析用户请求
    if isinstance(request, str):
        # 将字符串请求解析为命令和参数
        action, *args = parse_request_string(request)
    else:
        # 已经是结构化请求
        action, *args = request
    
    # 步骤3: 验证参数安全性
    if not validate_user_args(args):
        log_error("参数验证失败")
        return False
    
    # 步骤4: 更新系统设置
    args = update_settings_from_args(args)
    
    # 步骤5: 应用语言偏好
    apply_language_preference()
    
    # 步骤6: 选择合适的Agent
    action = normalize_action(action)
    if action not in command2class:
        log_warning("未知命令")
        return False
    
    # 步骤7: 执行通知回调(如果存在)
    if notify:
        notify()
    
    # 步骤8: 根据命令类型执行不同逻辑
    if action == "answer":
        # 回答模式特殊处理
        await PRReviewer(pr_url, is_answer=True, args=args, ai_handler=self.ai_handler).run()
    elif action == "auto_review":
        # 自动审查模式特殊处理
        await PRReviewer(pr_url, is_auto=True, args=args, ai_handler=self.ai_handler).run()
    else:
        # 标准命令处理
        await command2class[action](pr_url, ai_handler=self.ai_handler, args=args).run()
    
    # 步骤9: 返回成功状态
    return True
```

这个调度器实现了一个简洁而灵活的命令处理流程，通过以下几个关键机制实现高效调度：

1. **命令解析与规范化**：将用户输入转换为标准化的命令和参数
2. **参数验证与安全检查**：确保用户输入不会导致安全问题
3. **动态Agent选择**：基于命令类型选择合适的专业Agent
4. **上下文准备**：应用适当的设置和配置
5. **异步执行**：使用`await`关键字实现非阻塞执行

这种设计使得系统可以灵活处理各种命令，同时保持代码的可维护性和可扩展性。

## 7. 多Agent架构的优势与挑战

通过对PR-Agent多Agent系统架构的深入分析，可以总结出以下优势与挑战：

### 7.1 架构优势

1. **模块化与可扩展性**：每个Agent负责特定功能，使系统易于扩展和维护
2. **职责分离**：明确的职责边界减少了组件间的耦合
3. **配置驱动**：大量使用配置而非硬编码，提高了系统灵活性
4. **抽象层设计**：如AI处理器的抽象接口，使系统可以适应不同的后端服务
5. **异步处理**：基于`async/await`的设计提高了系统响应性和吞吐量

### 7.2 架构挑战

1. **状态管理复杂性**：多Agent间共享状态需要谨慎管理，避免不一致
2. **错误传播**：一个Agent的失败可能影响整个处理流程
3. **测试难度**：多Agent交互增加了测试的复杂性
4. **配置爆炸**：过度依赖配置可能导致系统行为难以预测
5. **性能开销**：Agent间通信和协调可能引入额外开销

PR-Agent通过精心的设计在很大程度上缓解了这些挑战，但在系统进一步扩展时，这些问题可能需要更系统化的解决方案。

## 8. 总结与架构评估

PR-Agent的多Agent系统架构展示了一种平衡灵活性和复杂性的设计方法。通过明确的职责分工、统一的接口设计、配置驱动的行为定制和强大的错误处理机制，系统实现了高度模块化和可扩展的代码审查平台。

特别值得注意的是系统对prompt管理的精细设计，这使得PR-Agent能够有效地引导AI模型生成高质量、结构化的输出，这是系统成功的关键因素之一。

从架构演进的角度看，PR-Agent当前的设计为未来的扩展和优化提供了良好的基础，但随着功能的增加和用户需求的变化，系统可能需要考虑更分布式的架构、更强大的状态管理机制和更精细的资源控制策略。
# PR-Agent 工具链实现原理分析

## 1. 工具抽象层设计

PR-Agent采用了一种高度模块化的工具抽象层设计，通过统一接口和适配器模式，成功地整合了Git操作、代码分析、AI模型调用等异构工具。这种设计不仅提高了系统的可维护性和可扩展性，还使得各组件能够独立演进而不影响整体架构。

### 1.1 核心抽象接口

PR-Agent的工具抽象层核心是一系列抽象接口，定义了各类工具的标准行为。通过分析源码，可以识别出以下几个关键抽象：

#### 1.1.1 Git提供商抽象

`GitProvider`抽象类（位于`pr_agent/git_providers/git_provider.py`）定义了与不同Git平台（如GitHub、GitLab、BitBucket）交互的统一接口：

```python
class GitProvider(ABC):
    @abstractmethod
    def get_diff_files(self) -> List[FilePatchInfo]:
        """获取PR的差异文件列表"""
        pass
    
    @abstractmethod
    def get_pr_description(self, split_changes_walkthrough=False) -> Tuple[str, List[str]]:
        """获取PR描述"""
        pass
    
    @abstractmethod
    def publish_comment(self, comment: str, is_temporary: bool = False) -> None:
        """发布评论"""
        pass
    
    # 更多抽象方法...
```

这种抽象使得系统可以无缝支持多种Git平台，而不需要修改上层业务逻辑。

#### 1.1.2 AI处理器抽象

`BaseAiHandler`抽象类（位于`pr_agent/algo/ai_handlers/base_ai_handler.py`）定义了与AI模型交互的标准接口：

```python
class BaseAiHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @property
    @abstractmethod
    def deployment_id(self):
        pass
    
    @abstractmethod
    async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2, img_path: str = None):
        """执行聊天完成请求"""
        pass
```

这种设计使得PR-Agent可以轻松切换不同的AI提供商（如OpenAI、Anthropic、Deepseek），而不影响核心功能。

#### 1.1.3 工具执行抽象

每个专业Agent工具类都实现了一个统一的执行接口模式，通常包含`run()`方法作为主要入口点：

```python
class PRTool:
    def __init__(self, pr_url: str, ai_handler: BaseAiHandler, args: list = None):
        # 初始化逻辑...
    
    async def run(self):
        # 执行逻辑...
```

这种一致的接口设计使得中央调度器可以统一管理和执行不同的工具。

### 1.2 适配器模式实现

PR-Agent广泛使用适配器模式来整合不同的外部工具和服务。以Git提供商为例，系统为每个支持的平台实现了专门的适配器类：

- `GitHubProvider`：适配GitHub API
- `GitLabProvider`：适配GitLab API
- `BitbucketProvider`：适配Bitbucket API
- `LocalGitProvider`：适配本地Git仓库

这些适配器类都继承自`GitProvider`抽象类，并实现了其定义的接口方法，但内部实现细节各不相同，以适应不同平台的特性。

工厂函数`get_git_provider`和`get_git_provider_with_context`负责根据URL或配置创建合适的Git提供商实例：

```python
def get_git_provider(pr_url: str) -> GitProvider:
    """
    根据PR URL创建合适的Git提供商实例
    """
    if pr_url.startswith("https://github.com"):
        return GitHubProvider(pr_url)
    elif pr_url.startswith("https://gitlab.com"):
        return GitLabProvider(pr_url)
    # 更多平台判断...
```

这种工厂模式与适配器模式的结合使得系统可以在运行时动态选择合适的实现，而不需要修改调用代码。

### 1.3 插件式架构

PR-Agent的工具抽象层还体现了插件式架构的特点，特别是在命令处理方面。系统通过`command2class`字典将命令字符串映射到对应的工具类：

```python
command2class = {
    "review": PRReviewer,
    "describe": PRDescription,
    "improve": PRCodeSuggestions,
    # 更多命令映射...
}
```

这种设计使得添加新工具只需两步：
1. 实现新的工具类，遵循统一接口
2. 在`command2class`字典中添加新的映射

这种松耦合的插件式架构大大降低了系统的维护成本和扩展难度。

## 2. 执行沙箱与安全策略

PR-Agent采用多层次的执行沙箱和安全策略，确保工具执行的隔离性和安全性。虽然系统没有使用传统意义上的容器化沙箱（如Docker或nsjail），但通过其他机制实现了有效的隔离和安全控制。

### 2.1 临时工作目录隔离

PR-Agent为每个操作创建和使用临时工作目录，这提供了一种基本的文件系统隔离：

```python
def create_temp_working_dir():
    """创建临时工作目录"""
    temp_dir = tempfile.mkdtemp()
    return temp_dir

def cleanup_temp_working_dir(temp_dir):
    """清理临时工作目录"""
    shutil.rmtree(temp_dir, ignore_errors=True)
```

这种方法确保了不同操作之间的文件隔离，防止意外的文件交叉污染。

### 2.2 参数验证与安全检查

PR-Agent实现了严格的参数验证机制，防止不安全的用户输入：

```python
def validate_user_args(args):
    """验证用户提供的参数安全性"""
    forbidden_args = get_settings().config.forbidden_cli_args
    for arg in args:
        if any(forbidden in arg for forbidden in forbidden_args):
            return False, arg
    return True, None
```

这种验证机制防止了命令注入和其他常见的安全漏洞。

### 2.3 资源限制与超时控制

PR-Agent对资源密集型操作实施了限制，特别是在处理大型PR时：

```python
def process_with_timeout(func, timeout, *args, **kwargs):
    """带超时的函数执行"""
    result = [None]
    exception = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # 超时处理
        thread.join()  # 等待线程完成
        raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]
```

这种超时控制机制防止了长时间运行的操作阻塞系统或消耗过多资源。

### 2.4 API访问控制

PR-Agent对外部API的访问实施了严格的控制，包括速率限制和错误处理：

```python
@retry(
    retry=(
        retry_if_exception_type(openai.error.Timeout) |
        retry_if_exception_type(openai.error.APIError) |
        retry_if_exception_type(openai.error.APIConnectionError) |
        retry_if_exception_type(openai.error.RateLimitError)
    ),
    stop=stop_after_attempt(OPENAI_RETRIES),
)
async def chat_completion(self, model: str, system: str, user: str, temperature: float = 0.2):
    # API调用实现...
```

这种设计不仅提高了系统的稳定性，还防止了API滥用和相关的安全风险。

### 2.5 权限分离与最小权限原则

PR-Agent遵循最小权限原则，特别是在处理Git操作时。系统只请求完成特定任务所需的最小权限集：

```python
def get_required_permissions():
    """获取所需的最小权限集"""
    permissions = {
        "contents": "read",
        "pull_requests": "write",
    }
    
    if get_settings().config.enable_commit_suggestions:
        permissions["contents"] = "write"
    
    return permissions
```

这种权限分离策略减少了潜在的安全风险，即使某个组件被攻击，其能够造成的损害也是有限的。

## 3. 结果解析器实现

PR-Agent的结果解析器是系统的关键组件之一，负责从各种工具输出中提取结构化数据。系统采用了多种技术来实现这一功能，包括正则表达式、JSON解析和AI辅助提取。

### 3.1 Git差异解析

PR-Agent实现了复杂的Git差异解析逻辑，能够从原始补丁中提取结构化信息：

```python
def process_patch_lines(patch_str, original_file_str, patch_extra_lines_before, patch_extra_lines_after, new_file_str=""):
    """处理补丁行，提取结构化信息"""
    file_original_lines = original_file_str.splitlines()
    file_new_lines = new_file_str.splitlines() if new_file_str else []
    patch_lines = patch_str.splitlines()
    extended_patch_lines = []
    
    # 使用正则表达式识别补丁块头
    RE_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[ ]?(.*)")
    
    # 处理每一行
    for i, line in enumerate(patch_lines):
        if line.startswith('@@'):
            match = RE_HUNK_HEADER.match(line)
            if match:
                # 提取补丁块信息
                section_header, size1, size2, start1, start2 = extract_hunk_headers(match)
                # 处理补丁块...
```

这种基于正则表达式的解析方法能够高效地处理Git差异格式，提取出文件名、变更类型、行号等关键信息。

### 3.2 JSON响应解析

PR-Agent广泛使用JSON格式作为AI模型响应的结构化数据格式。
(Content truncated due to size limit. Use line ranges to read in chunks)