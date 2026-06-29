import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Load report
# --------------------------------------------------

with open("param_search_results/param_search_report.json", "r") as f:
    report = json.load(f)

results = report["all_results"]

# --------------------------------------------------
# Build graph
# --------------------------------------------------

G = nx.Graph()

features = []
node_ids = []

for i, r in enumerate(results):
    combo = r["combo"]

    node_id = f"{r['emotion']}_{i}"

    G.add_node(
        node_id,
        composite=r["composite"],
        emotion=r["emotion"],
        temperature=combo["TEMPERATURE"],
        top_p=combo["TOP_P"],
        repetition_penalty=combo["REPETITION_PENALTY"],
        max_tokens=combo["MAX_TOKENS_PER_PANEL"],
    )

    features.append([
        combo["TEMPERATURE"],
        combo["TOP_P"],
        combo["REPETITION_PENALTY"],
        combo["MAX_TOKENS_PER_PANEL"] / 250,
        r["composite"] / 100,
        r["rule_pass_rate"] / 100,
        r["hallucination_rate"] / 100,
    ])

    node_ids.append(node_id)

features = np.array(features)

# --------------------------------------------------
# Similarity edges
# --------------------------------------------------

sim = cosine_similarity(features)

THRESHOLD = 0.85

for i in range(len(node_ids)):
    for j in range(i + 1, len(node_ids)):
        if sim[i, j] > THRESHOLD:
            G.add_edge(
                node_ids[i],
                node_ids[j],
                weight=float(sim[i, j])
            )

print(
    f"Graph built: "
    f"{G.number_of_nodes()} nodes, "
    f"{G.number_of_edges()} edges"
)

# --------------------------------------------------
# Visualization
# --------------------------------------------------

plt.figure(figsize=(16, 12))

pos = nx.spring_layout(
    G,
    seed=42,
    k=1.2 / np.sqrt(max(len(G.nodes()), 1))
)

scores = np.array([
    G.nodes[n]["composite"]
    for n in G.nodes()
])

sizes = 100 + scores * 12

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.15,
    width=0.5
)

nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_color=scores,
    cmap="RdYlGn",
    node_size=sizes
)

# label top performers only
top_nodes = sorted(
    G.nodes(),
    key=lambda x: G.nodes[x]["composite"],
    reverse=True
)[:10]

labels = {
    n: f"{G.nodes[n]['emotion']}\n{G.nodes[n]['composite']:.1f}"
    for n in top_nodes
}

nx.draw_networkx_labels(
    G,
    pos,
    labels,
    font_size=8
)

plt.colorbar(nodes, label="Composite Score")
plt.title("MoodWeaver Parameter Knowledge Graph")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "knowledge_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()