"""
DSP Case Study plotting script
Place this script in the same directory as the supplied CSV files and run:
    python make_report_plots.py

Required files:
  baseline_metrics.csv, clap_test_predictions.csv, clap_metrics.csv,
  clap_classification_report.csv, clap_confusion_matrix.csv,
  clap_noise_predictions.csv, clap_noise_summary.csv,
  clap_audio_dependence_summary.csv

The baseline confusion matrix is encoded from the supplied baseline plot:
rows/columns = car_horn, chainsaw, clock_tick, dog, rain.
All output figures are saved to ./report_plots/.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_plots"
OUT.mkdir(exist_ok=True)

pred = pd.read_csv(ROOT / "clap_test_predictions.csv")
noise = pd.read_csv(ROOT / "clap_noise_summary.csv")
noise_pred = pd.read_csv(ROOT / "clap_noise_predictions.csv")
dep = pd.read_csv(ROOT / "clap_audio_dependence_summary.csv")
clap_cm_df = pd.read_csv(ROOT / "clap_confusion_matrix.csv", index_col=0)
report = pd.read_csv(ROOT / "clap_classification_report.csv", index_col=0)
base_metrics = pd.read_csv(ROOT / "baseline_metrics.csv").iloc[0]

classes = ["car_horn", "chainsaw", "clock_tick", "dog", "rain"]
baseline_cm = np.array([[5,1,1,1,0],[1,2,1,0,4],[0,2,2,4,0],[0,0,0,8,0],[0,3,0,0,5]])
clap_cm = clap_cm_df.loc[classes, classes].values
def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close()

fig, axes = plt.subplots(1,2,figsize=(13,5.6))
for ax,cm,title in zip(axes,[baseline_cm,clap_cm],["MFCC + SVM baseline (55%)","CLAP zero-shot (100%)"]):
    im=ax.imshow(cm,cmap="Blues")
    ax.set_title(title,fontweight="bold")
    ax.set_xticks(range(5),classes,rotation=35,ha="right")
    ax.set_yticks(range(5),classes)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    for i in range(5):
        for j in range(5):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
fig.suptitle("Confusion Matrix Comparison — 40 ESC-50 Test Clips",fontweight="bold")
save("01_confusion_matrices_comparison.png")

acc=[float(base_metrics["TestAccuracy"]),float(pred["correct"].mean())]
f1=[float(base_metrics["MacroF1"]),float(report.loc["macro avg","f1-score"])]
x=np.arange(2); w=.34
plt.figure(figsize=(8,5))
plt.bar(x-w/2,np.array(acc)*100,w,label="Accuracy (%)")
plt.bar(x+w/2,np.array(f1)*100,w,label="Macro-F1 (%)")
plt.xticks(x,["MFCC + SVM","CLAP zero-shot"]); plt.ylabel("Score (%)"); plt.ylim(0,110)
plt.title("Baseline vs. Pretrained Audio–Text Model"); plt.legend()
for i,v in enumerate(acc): plt.text(i-w/2,v*100+1,f"{v*100:.1f}%",ha="center")
for i,v in enumerate(f1): plt.text(i+w/2,v*100+1,f"{v*100:.1f}%",ha="center")
save("02_model_metrics_comparison.png")

recall_base=np.diag(baseline_cm)/baseline_cm.sum(axis=1)
recall_clap=report.loc[classes,"recall"].values
x=np.arange(5); w=.36
plt.figure(figsize=(9,5))
plt.bar(x-w/2,recall_base*100,w,label="MFCC + SVM")
plt.bar(x+w/2,recall_clap*100,w,label="CLAP zero-shot")
plt.xticks(x,classes,rotation=25,ha="right"); plt.ylabel("Recall (%)"); plt.ylim(0,110)
plt.title("Per-Class Recall Comparison"); plt.legend(); save("03_per_class_recall.png")

metrics=report.loc[classes,["precision","recall","f1-score"]]
x=np.arange(5); w=.24
plt.figure(figsize=(9,5))
for k,col in enumerate(["precision","recall","f1-score"]):
    plt.bar(x+(k-1)*w,metrics[col].values,w,label=col.replace("-score","").title())
plt.xticks(x,classes,rotation=25,ha="right"); plt.ylabel("Score"); plt.ylim(0,1.12)
plt.title("CLAP Per-Class Precision, Recall and F1"); plt.legend(); save("04_clap_per_class_metrics.png")

noise=noise.set_index("condition").loc[["clean","snr_10dB","snr_5dB"]].reset_index()
labels=["Clean","10 dB SNR","5 dB SNR"]
plt.figure(figsize=(8,5))
plt.plot(labels,noise["accuracy_percent"],marker="o",linewidth=2); plt.ylim(0,110)
plt.ylabel("Accuracy (%)"); plt.xlabel("Audio condition")
plt.title("CLAP Accuracy Under Additive Gaussian Noise")
for i,v in enumerate(noise["accuracy_percent"]): plt.annotate(f"{v:.0f}%",(i,v),xytext=(0,8),textcoords="offset points",ha="center")
save("05_noise_robustness_accuracy.png")

plt.figure(figsize=(8,5))
plt.plot(labels,noise["mean_similarity"],marker="o",linewidth=2)
plt.ylabel("Mean top-class similarity"); plt.xlabel("Audio condition")
plt.title("Effect of Noise on CLAP Similarity Scores")
for i,v in enumerate(noise["mean_similarity"]): plt.annotate(f"{v:.4f}",(i,v),xytext=(0,8),textcoords="offset points",ha="center")
save("06_noise_similarity.png")

noise_pred["condition_label"]=noise_pred["condition"].map({"clean":"Clean","snr_10dB":"10 dB SNR","snr_5dB":"5 dB SNR"})
order=["Clean","10 dB SNR","5 dB SNR"]
plt.figure(figsize=(8,5))
plt.boxplot([noise_pred.loc[noise_pred.condition_label==c,"max_similarity"].dropna().values for c in order],tick_labels=order,showmeans=True)
plt.ylabel("Winning text–audio similarity"); plt.xlabel("Audio condition")
plt.title("Distribution of CLAP Winning Similarity Scores"); save("07_similarity_score_distribution.png")

dep=dep.set_index("condition").loc[["correct_audio","shuffled_audio"]].reset_index()
vals=dep["accuracy"].values*100
plt.figure(figsize=(8,5)); bars=plt.bar(["Correct audio-label pairing","Shuffled audio"],vals)
plt.ylabel("Accuracy (%)"); plt.ylim(0,110); plt.title("Audio-Dependence Control")
plt.xticks(rotation=10,ha="right")
for b,v in zip(bars,vals): plt.text(b.get_x()+b.get_width()/2,v+1,f"{v:.1f}%",ha="center")
plt.axhline(20,linestyle="--",linewidth=1.5,label="Uniform random baseline (20%)"); plt.legend()
save("08_audio_dependence_control.png")

sim_cols={"dog":"similarity_dog","rain":"similarity_rain","car_horn":"similarity_car_horn","chainsaw":"similarity_chainsaw","clock_tick":"similarity_clock_tick"}
mean_sims=pred.groupby("category")[[sim_cols[c] for c in classes]].mean().loc[classes]
mean_sims.columns=classes
plt.figure(figsize=(8,5.5)); im=plt.imshow(mean_sims.values,cmap="viridis",aspect="auto")
plt.xticks(range(5),classes,rotation=30,ha="right"); plt.yticks(range(5),classes)
plt.xlabel("Candidate text label"); plt.ylabel("True sound category")
plt.title("Mean CLAP Audio–Text Similarity by True Class")
for i in range(5):
    for j in range(5): plt.text(j,i,f"{mean_sims.iloc[i,j]:.2f}",ha="center",va="center",color="white" if mean_sims.iloc[i,j]<np.nanmean(mean_sims.values) else "black",fontsize=9)
plt.colorbar(im,label="Cosine similarity"); save("09_class_similarity_heatmap.png")

sim=pred[[sim_cols[c] for c in classes]].to_numpy()
sorted_sim=np.sort(sim,axis=1); margin=sorted_sim[:,-1]-sorted_sim[:,-2]
plt.figure(figsize=(8,5)); plt.hist(margin,bins=10,edgecolor="black")
plt.xlabel("Top-1 minus top-2 similarity"); plt.ylabel("Number of test clips")
plt.title("CLAP Prediction Similarity Margin (40 Test Clips)"); save("10_similarity_margin_histogram.png")

pd.DataFrame({"Model":["MFCC + SVM","CLAP zero-shot"],"Test clips":[40,40],
              "Accuracy (%)":[acc[0]*100,acc[1]*100],"Macro-F1 (%)":[f1[0]*100,f1[1]*100]}).to_csv(OUT/"derived_model_comparison.csv",index=False)
print(f"Saved 10 figures and a comparison CSV to: {OUT}")
