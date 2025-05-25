import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.patches as mpatches

# 创建有向图
G = nx.DiGraph()

# 添加节点
nodes = [
    ("输入", "PR差异文件\n和补丁内容", 0),
    ("步骤1", "文件预处理与排序", 1),
    ("步骤2", "移除纯删除块", 2),
    ("步骤3", "计算Token消耗", 3),
    ("步骤4", "按优先级迭代填充", 4),
    ("步骤5", "检查Token预算", 5),
    ("决策", "是否超出\nToken限制?", 6),
    ("步骤6A", "添加文件到\n最终补丁列表", 7),
    ("步骤6B", "停止添加更多文件", 8),
    ("步骤7", "组合最终差异字符串", 9),
    ("步骤8", "添加省略文件信息", 10),
    ("输出", "压缩后的PR差异", 11)
]

# 添加节点到图中
for i, (node_type, label, level) in enumerate(nodes):
    if node_type == "决策":
        node_color = "lightyellow"
        node_shape = "diamond"
    elif node_type == "输入" or node_type == "输出":
        node_color = "lightblue"
        node_shape = "box"
    else:
        node_color = "lightgreen"
        node_shape = "ellipse"
    
    G.add_node(i, label=label, level=level, node_color=node_color, node_shape=node_shape)

# 添加边
edges = [
    (0, 1),  # 输入 -> 步骤1
    (1, 2),  # 步骤1 -> 步骤2
    (2, 3),  # 步骤2 -> 步骤3
    (3, 4),  # 步骤3 -> 步骤4
    (4, 5),  # 步骤4 -> 步骤5
    (5, 6),  # 步骤5 -> 决策
    (6, 7),  # 决策 -> 步骤6A (否)
    (6, 8),  # 决策 -> 步骤6B (是)
    (7, 4),  # 步骤6A -> 步骤4 (循环)
    (7, 9),  # 步骤6A -> 步骤7
    (8, 9),  # 步骤6B -> 步骤7
    (9, 10), # 步骤7 -> 步骤8
    (10, 11) # 步骤8 -> 输出
]

# 添加边到图中
for src, dst in edges:
    G.add_edge(src, dst)

# 设置节点位置
pos = {}
for node, data in G.nodes(data=True):
    level = data['level']
    if node == 6:  # 决策节点
        pos[node] = (0.5, -level)
    elif node == 7:  # 步骤6A
        pos[node] = (0, -level)
    elif node == 8:  # 步骤6B
        pos[node] = (1, -level)
    else:
        pos[node] = (0.5, -level)

# 创建图形
plt.figure(figsize=(10, 14))

# 绘制节点
for node, (x, y) in pos.items():
    node_shape = G.nodes[node]['node_shape']
    node_color = G.nodes[node]['node_color']
    label = G.nodes[node]['label']
    
    if node_shape == "diamond":
        plt.scatter(x, y, s=3000, marker='D', color=node_color, edgecolors='black', zorder=2)
    elif node_shape == "box":
        plt.scatter(x, y, s=3000, marker='s', color=node_color, edgecolors='black', zorder=2)
    else:  # ellipse
        plt.scatter(x, y, s=3000, marker='o', color=node_color, edgecolors='black', zorder=2)
    
    plt.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', zorder=3)

# 绘制边
for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    
    # 特殊处理循环边
    if u == 7 and v == 4:  # 步骤6A -> 步骤4 (循环)
        plt.annotate("", xy=(0.2, -4), xytext=(0, -7),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                   connectionstyle="arc3,rad=-0.3"), zorder=1)
        plt.text(0.1, -5.5, "继续下一个文件", fontsize=8, ha='center', rotation=90)
    else:
        # 为决策节点添加标签
        if u == 6:
            if v == 7:  # 否
                plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5), zorder=1)
                plt.text((x1+x2)/2 - 0.1, (y1+y2)/2, "否", fontsize=10, ha='right')
            elif v == 8:  # 是
                plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5), zorder=1)
                plt.text((x1+x2)/2 + 0.1, (y1+y2)/2, "是", fontsize=10, ha='left')
        else:
            plt.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5), zorder=1)

# 添加图例
input_patch = mpatches.Patch(color='lightblue', label='输入/输出')
step_patch = mpatches.Patch(color='lightgreen', label='处理步骤')
decision_patch = mpatches.Patch(color='lightyellow', label='决策点')
plt.legend(handles=[input_patch, step_patch, decision_patch], loc='upper center', bbox_to_anchor=(0.5, 0.05))

# 添加标题
plt.title("PR-Agent 差异压缩算法流程图", fontsize=16, fontweight='bold', pad=20)

# 调整布局
plt.axis('off')
plt.tight_layout()

# 保存图片
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'pr_agent_compression_algorithm.png'), dpi=300, bbox_inches='tight')
plt.close()

print("算法流程图已生成并保存为 images/pr_agent_compression_algorithm.png")
