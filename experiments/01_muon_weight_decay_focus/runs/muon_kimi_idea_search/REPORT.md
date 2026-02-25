# Muon Kimi-Style Idea Search Report (Single Seed)

mode | wd | val_loss | train_loss | val_acc | delta_vs_baseline
---|---:|---:|---:|---:|---:
update | -0.2 | 7.2825 | 7.2916 | 0.1301 | -0.0070
param | -0.01 | 7.2872 | 7.2979 | 0.1298 | -0.0023
update | -0.05 | 7.2879 | 7.2961 | 0.1289 | -0.0016
param | 0.05 | 7.2880 | 7.2984 | 0.1296 | -0.0015
both | 0.01 | 7.2891 | 7.2990 | 0.1296 | -0.0004
param | 0.0 | 7.2895 | 7.3012 | 0.1295 | +0.0000
param | -0.05 | 7.2902 | 7.3025 | 0.1291 | +0.0007
both | -0.01 | 7.2902 | 7.3016 | 0.1290 | +0.0007
param | 0.01 | 7.2902 | 7.3013 | 0.1293 | +0.0007
update | 0.05 | 7.2940 | 7.3033 | 0.1290 | +0.0045
update | 0.2 | 7.2977 | 7.3044 | 0.1266 | +0.0082

Best idea: mode=`update`, wd=`-0.2` with val_loss=`7.2825`
Baseline (param, wd=0): `7.2895`
Best delta vs baseline: `-0.0070`

![Ranked val loss](./ranked_val_loss.png)

![Delta vs baseline](./delta_vs_baseline.png)
