"""Professional result plots from frozen prediction metrics, without model changes."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR",str(ROOT / ".runtime/mpl"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    m=pd.read_csv(ROOT / "results/metrics.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.titleweight":"bold","figure.dpi":150})
    tasks=[("detection","recall","Fault detection recall"),("classification","accuracy","Fault-type accuracy"),("localization","exact_bus_accuracy","Exact-bus accuracy")]
    out=ROOT / "reports/figures"
    for stratum,name,xlabel in [("Scenario_ID","der_performance","DER configuration (joint PV / EV / V2G changes)"),("resistance_bin","resistance_performance","Fault resistance (ohm)")]:
        fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
        for ax,(task,metric,title) in zip(axs,tasks):
            d=m[(m.experiment=="full")&(m.partition=="test")&(m.stratum==stratum)&(m.task==task)&(m.metric==metric)]
            if stratum=="Scenario_ID": labels=sorted(d.value.unique())
            else: labels=["(-0.001, 1.0]","(1.0, 5.0]","(5.0, 10.0]","(10.0, 15.0]","(15.0, 25.0]"]
            for model,color,marker in [("B0","#9aa6af","^"),("B1","#d47437","s"),("LightGBM","#087f8c","o")]:
                z=d[d.model.eq(model)].set_index("value").result.reindex(labels)
                if z.notna().any(): ax.plot(range(len(labels)),z*100,label=model,color=color,marker=marker,lw=1.8)
            ax.set_xticks(range(len(labels)),labels,rotation=30 if stratum=="resistance_bin" else 0,ha="right" if stratum=="resistance_bin" else "center"); ax.set_title(title); ax.set_ylim(0,103); ax.grid(axis="y",alpha=.2)
        axs[0].set_ylabel("Held-out score (%)"); axs[-1].legend(loc="lower left"); fig.supxlabel(xlabel); fig.tight_layout(); fig.savefig(out / f"{name}.png",bbox_inches="tight"); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    tags=["full","available_93","available_36","available_21","available_9"]
    for ax,metric,title,scale in [(axs[0],"exact_bus_accuracy","Exact-bus accuracy (%) — higher is better",100),(axs[1],"mean_branch_hops","Mean network branch hops — lower is better",1)]:
        for offset,model,color in [(-.18,"B1","#d47437"),(.18,"LightGBM","#087f8c")]:
            d=m[(m.partition=="test")&(m.stratum=="all")&(m.task=="localization")&(m.metric==metric)&(m.model==model)].set_index("experiment").result
            ax.bar(np.arange(len(tags))+offset,d.reindex(tags)*scale,width=.35,label=model,color=color)
        ax.set_xticks(range(len(tags)),["Released\n901 channels","93 local\nPMUs","36 available\nPMUs","21 available\nPMUs","9 available\nPMUs"])
        ax.set_title(title); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.text(.5,.005,"Placements differ and are not nested; this is not a controlled sensor-count-only sweep.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out / "sensor_localization_performance.png",bbox_inches="tight"); plt.close(fig)
    print("Result plots generated",flush=True)


if __name__=="__main__": main()
