import matplotlib.pyplot as plt
import networkx as nx
import os
import math  # 导入math库用于三角函数

# 创建有向图
G = nx.DiGraph()

# 添加节点
# 中央调度器
G.add_node("PRAgent\n(中央调度器)", pos=(0, 0), node_color='lightblue', node_size=3000)

# 专业Agent节点
agents = [
    "PRReviewer\n(代码审查)", 
    "PRDescription\n(PR描述生成)", 
    "PRCodeSuggestions\n(代码改进)",
    "PRQuestions\n(问题回答)",
    "PRUpdateChangelog\n(更新日志)",
    "PRSimilarIssue\n(相似问题)",
    "PRGenerateLabels\n(标签生成)"
]

# 添加专业Agent节点
angle_step = 2 * math.pi / len(agents)
radius = 5
for i, agent in enumerate(agents):
    angle = i * angle_step
    x = radius * 1.2 * math.cos(angle)  # 使用math.cos替代nx.cos
    y = radius * math.sin(angle)  # 使用math.sin替代nx.sin
    G.add_node(agent, pos=(x, y), node_color='lightgreen', node_size=2500)
    # 添加从中央调度器到专业Agent的边
    G.add_edge("PRAgent\n(中央调度器)", agent, edge_color='black')

# 添加基础设施节点
infra_nodes = [
    "Git提供商抽象\n(GitProvider)", 
    "AI处理器抽象\n(BaseAiHandler)",
    "Token管理器\n(TokenHandler)",
    "配置系统\n(Settings)"
]

# 添加基础设施节点
infra_pos_y = -8
for i, node in enumerate(infra_nodes):
    pos_x = -6 + i * 4
    G.add_node(node, pos=(pos_x, infra_pos_y), node_color='lightyellow', node_size=2500)
    # 添加从中央调度器到基础设施的边
    G.add_edge("PRAgent\n(中央调度器)", node, edge_color='blue')
    # 添加从专业Agent到基础设施的边
    for agent in agents:
        G.add_edge(agent, node, edge_color='gray', style='dashed')

# 添加具体实现节点
impl_nodes = {
    "Git提供商抽象\n(GitProvider)": ["GitHubProvider", "GitLabProvider", "BitbucketProvider"],
    "AI处理器抽象\n(BaseAiHandler)": ["LiteLLMAIHandler", "OpenAIAIHandler", "AnthropicAIHandler"]
}

# 添加具体实现节点
impl_pos_y = -12
for base_node, impls in impl_nodes.items():
    base_x = list(nx.get_node_attributes(G, 'pos')[base_node])[0]
    for i, impl in enumerate(impls):
        offset = (i - (len(impls) - 1) / 2) * 2
        G.add_node(impl, pos=(base_x + offset, impl_pos_y), node_color='lightpink', node_size=1500)
        G.add_edge(base_node, impl, edge_color='red')

# 获取节点位置
pos = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(16, 12))
# 绘制节点
node_colors = [data.get('node_color', 'lightblue') for _, data in G.nodes(data=True)]
node_sizes = [data.get('node_size', 2000) for _, data in G.nodes(data=True)]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

# 绘制边
edge_colors = [G.edges[edge].get('edge_color', 'black') for edge in G.edges()]
edge_styles = [G.edges[edge].get('style', 'solid') for edge in G.edges()]
solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style', 'solid') == 'solid']
dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style', 'solid') == 'dashed']

nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color=[G.edges[edge].get('edge_color', 'black') for edge in solid_edges], arrows=True)
nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color=[G.edges[edge].get('edge_color', 'gray') for edge in dashed_edges], style='dashed', arrows=True)

# 绘制标签
nx.draw_networkx_labels(G, pos, font_size=10, font_family='SimHei')

# 添加图例
plt.text(-10, 3, "节点类型:", fontsize=12, fontweight='bold')
plt.text(-10, 2, "■ 中央调度器", fontsize=10, color='lightblue')
plt.text(-10, 1, "■ 专业Agent", fontsize=10, color='lightgreen')
plt.text(-10, 0, "■ 基础设施抽象", fontsize=10, color='lightyellow')
plt.text(-10, -1, "■ 具体实现", fontsize=10, color='lightpink')

plt.text(5, 3, "连接类型:", fontsize=12, fontweight='bold')
plt.text(5, 2, "— 调度控制", fontsize=10, color='black')
plt.text(5, 1, "— 基础设施依赖", fontsize=10, color='blue')
plt.text(5, 0, "— 实现关系", fontsize=10, color='red')
plt.text(5, -1, "--- 使用关系", fontsize=10, color='gray')

plt.title("PR-Agent 多Agent系统架构图", fontsize=16, fontweight='bold')
plt.axis('off')

# 保存图片
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'pr_agent_architecture.png'), dpi=300, bbox_inches='tight')
plt.close()

print("架构图已生成并保存为 images/pr_agent_architecture.png")
