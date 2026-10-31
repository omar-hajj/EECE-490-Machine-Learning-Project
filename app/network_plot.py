"""Source-topology schematic layout, independent of geographical coordinates."""
from collections import deque
import matplotlib.pyplot as plt


def draw_network(topology,predicted=None,actual=None):
    graph={b:set() for b in topology["buses"]}
    for e in topology["edges"]: graph[e["u"]].add(e["v"]); graph[e["v"]].add(e["u"])
    root="eq_source_bus"; parent={root:None}; levels={root:0}; queue=deque([root]); children={b:[] for b in graph}
    while queue:
        u=queue.popleft()
        for v in sorted(graph[u]):
            if v not in parent: parent[v]=u; levels[v]=levels[u]+1; children[u].append(v); queue.append(v)
    positions={}; leaf=[0]
    def place(u):
        if not children[u]: y=leaf[0]; leaf[0]+=1
        else: y=sum(place(v) for v in children[u])/len(children[u])
        positions[u]=(levels[u],y); return y
    place(root)
    fig,ax=plt.subplots(figsize=(11,6))
    for u,neighbors in graph.items():
        for v in neighbors:
            if u<v: ax.plot([positions[u][0],positions[v][0]],[positions[u][1],positions[v][1]],color="#ced6da",lw=.6,zorder=1)
    ax.scatter([positions[b][0] for b in positions],[positions[b][1] for b in positions],s=6,color="#638c96",zorder=2)
    for bus,color,label,marker in [(actual,"#d47437","Actual fault","o"),(predicted,"#124bc0","LightGBM prediction","x")]:
        if bus in positions:
            x,y=positions[bus]; ax.scatter([x],[y],s=100,color=color,label=f"{label}: {bus}",marker=marker,zorder=4); ax.annotate(bus,(x,y),xytext=(8,6),textcoords="offset points",fontsize=9,color=color)
    ax.set_title("Verified source topology — schematic branch-depth layout")
    ax.set_xlabel("Branches from source (layout only)"); ax.set_ylabel("Schematic feeder position")
    if predicted or actual: ax.legend(loc="upper left",fontsize=8)
    ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
    return fig
