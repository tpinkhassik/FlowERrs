import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/ptim/orcd/scratch/FlowERrs_checkpoints/flower_new_dataset/best_large_hyperparam/loss_history.csv")
plt.plot(df["step"], df["cv_loss"], label="cv_loss")
plt.plot(df["step"], df["be_loss"], label="be_loss")
plt.legend()
plt.xlabel("step")
plt.ylabel("loss")
plt.savefig("loss_plot.pdf")
plt.close()