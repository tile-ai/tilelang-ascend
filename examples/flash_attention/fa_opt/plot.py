import matplotlib.pyplot as plt
import pandas as pd

file_path = ".log/run_cmp_performance_${RUN_ID}/comparison_report_${RUN_ID}.csv"

df = pd.read_csv(file_path)

df["TL_Time(ms)"] = df["TL_Time(us)"] / 1000.0
df["AC_Time(ms)"] = df["AC_Time(us)"] / 1000.0

x = df["S"]  # seqlen
plt.plot(x, df["TL_Time(ms)"], marker="o", label="TL-Ascend")
plt.plot(x, df["AC_Time(ms)"], marker="o", label="AscendC")

plt.title("TL-Ascend vs AscendC")
plt.xlabel("Seqlen (S)")
plt.ylabel("Time (ms)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("time_tl-ascend_vs_ascendc.png", dpi=200)
