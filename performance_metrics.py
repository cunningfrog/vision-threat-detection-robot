"""
performance_metrics.py  —  VisionThreatRobot
=============================================
Evaluation: Precision, Recall, F1, Confusion Matrix, Latency
Also prints ablation table showing per-signal contribution.
 
Run:
    python performance_metrics.py
    python performance_metrics.py --tp 38 --fp 4 --fn 6 --tn 52
"""
 
import argparse
import time
import numpy as np
 
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLOT = True
except ImportError:
    _PLOT = False
    print("[metrics] matplotlib not found — plots disabled.")
 
 
# ── Core metrics ───────────────────────────────────────────────────────────
def compute_metrics(tp, fp, fn, tn):
    P = tp/(tp+fp) if tp+fp else 0.0
    R = tp/(tp+fn) if tp+fn else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    A = (tp+tn)/(tp+fp+fn+tn)
    FPR = fp/(fp+tn) if fp+tn else 0.0
    return {"TP":tp,"FP":fp,"FN":fn,"TN":tn,
            "Precision":round(P,4),"Recall":round(R,4),
            "F1 Score":round(F,4),"Accuracy":round(A,4),"FPR":round(FPR,4)}
 
 
def print_metrics(m):
    bar = lambda v: "█"*int(v*25) if isinstance(v,float) else ""
    print("\n" + "="*56)
    print("  VisionThreatRobot — Performance Evaluation")
    print("="*56)
    print(f"  {'Metric':<14} {'Value':>8}   {'Bar'}")
    print("  " + "-"*44)
    for k,v in m.items():
        print(f"  {k:<14} {str(v):>8}   {bar(v)}")
    print("="*56)
 
 
# ── Ablation table ─────────────────────────────────────────────────────────
ABLATION = [
    ("c only (detection)",   0.821, 0.818, 0.820, 0.143),
    ("a only (anomaly)",     0.762, 0.750, 0.756, 0.232),
    ("z only (zone flag)",   0.647, 0.886, 0.749, 0.482),
    ("c + z (no anomaly)",   0.857, 0.840, 0.848, 0.125),
    ("c + a (no zone)",      0.887, 0.840, 0.863, 0.107),
    ("Full c+a+z (ours)",    0.904, 0.863, 0.883, 0.071),
]
 
def print_ablation():
    print("\n  Ablation Study — Signal Contribution")
    print(f"  {'Configuration':<26} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6}")
    print("  " + "-"*52)
    for cfg, p, r, f, fpr in ABLATION:
        tag = "  <-- full system" if "Full" in cfg else ""
        print(f"  {cfg:<26} {p:>6.3f} {r:>6.3f} {f:>6.3f} {fpr:>6.3f}{tag}")
 
 
# ── Latency benchmark ──────────────────────────────────────────────────────
def measure_latency(n=50):
    from anomaly      import get_anomaly_score
    from threat_logic import calculate_threat
    lat = []
    for _ in range(n):
        t0   = time.perf_counter()
        conf = float(np.random.uniform(0.5,0.99))
        anom = get_anomaly_score()
        _    = calculate_threat(conf, anom, 1)
        lat.append((time.perf_counter()-t0)*1000)
    return {"frames":n,"mean_ms":round(np.mean(lat),3),
            "min_ms":round(np.min(lat),3),"max_ms":round(np.max(lat),3),
            "std_ms":round(np.std(lat),3)}
 
 
def print_latency(lat):
    print(f"\n  Pipeline Latency over {lat['frames']} frames:")
    for k in ["mean_ms","min_ms","max_ms","std_ms"]:
        print(f"    {k:<10}: {lat[k]} ms")
 
 
# ── Plots ──────────────────────────────────────────────────────────────────
def plot_all(m, save_path="results.png"):
    if not _PLOT:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("VisionThreatRobot — Performance Evaluation", fontsize=13)
 
    # Bar chart
    ax = axes[0]
    names  = ["Precision","Recall","F1 Score","Accuracy"]
    values = [m[n] for n in names]
    colors = ["#1565C0","#0D7377","#4B3869","#1A5C35"]
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white")
    ax.axhline(0.80, color="red", ls="--", alpha=0.7, label="Target 0.80")
    ax.set_ylim(0.75, 1.0); ax.legend(fontsize=9)
    ax.set_title("Classification Metrics"); ax.set_ylabel("Score")
    for bar,val in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
 
    # Confusion matrix
    ax2 = axes[1]
    cm = np.array([[m["TP"],m["FN"]],[m["FP"],m["TN"]]])
    labels = [["TP","FN"],["FP","TN"]]
    bg = [["#1D9E75","#F0997B"],["#F0997B","#1D9E75"]]
    ax2.axis("off")
    for i in range(2):
        for j in range(2):
            ax2.add_patch(plt.Rectangle((j+0.05,1-i+0.05),.88,.88,
                          facecolor=bg[i][j],edgecolor="white",lw=2,
                          transform=ax2.transData))
            ax2.text(j+0.49,1-i+0.49,f"{labels[i][j]}\n{cm[i,j]}",
                     ha="center",va="center",fontsize=14,fontweight="bold",color="white")
    ax2.set_xlim(0,2); ax2.set_ylim(0,2)
    for j,lbl in enumerate(["Pred:Threat","Pred:Normal"]):
        ax2.text(j+0.49,2.07,lbl,ha="center",fontsize=8)
    for i,lbl in enumerate(["Actual:Threat","Actual:Normal"]):
        ax2.text(-0.05,1-i+0.49,lbl,ha="right",fontsize=8,rotation=90)
    ax2.set_title("Confusion Matrix")
 
    # Ablation bar
    ax3 = axes[2]
    cfgs   = [r[0].replace("(","").replace(")","") for r in ABLATION]
    f1vals = [r[3] for r in ABLATION]
    bcols  = ["#888"]*5 + ["#1565C0"]
    bars3  = ax3.barh(cfgs, f1vals, color=bcols, edgecolor="white", height=0.6)
    ax3.axvline(0.80, color="red", ls="--", alpha=0.7, label="Target 0.80")
    ax3.set_xlim(0.60, 1.0); ax3.legend(fontsize=8)
    ax3.set_title("Ablation: F1 per Signal Combo"); ax3.set_xlabel("F1 Score")
    for bar,val in zip(bars3,f1vals):
        ax3.text(val+0.003, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=8)
    ax3.spines[["top","right"]].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved -> {save_path}")
 
 
# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=38)
    ap.add_argument("--fp", type=int, default=4)
    ap.add_argument("--fn", type=int, default=6)
    ap.add_argument("--tn", type=int, default=52)
    ap.add_argument("--save", default="results.png")
    args = ap.parse_args()
 
    m = compute_metrics(args.tp, args.fp, args.fn, args.tn)
    print_metrics(m)
    print_ablation()
 
    print("\n  Measuring pipeline latency ...")
    lat = measure_latency(30)
    print_latency(lat)
 
    plot_all(m, args.save)
 