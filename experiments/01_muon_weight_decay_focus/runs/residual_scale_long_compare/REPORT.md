# Residual-Scale Long Comparison (20M Tokens)

residual_scale | val_loss | train_loss | val_acc | tokens_seen | steps | delta_vs_scale1
---|---:|---:|---:|---:|---:|---:
0.69 | 4.5363 | 4.4171 | 0.2751 | 20004864 | 1221 | -0.0072
1.0 | 4.5435 | 4.4266 | 0.2749 | 20004864 | 1221 | +0.0000

Baseline (scale=1.0) val_loss: `4.5435`\

Best candidate (scale=0.69) val_loss: `4.5363`\

Delta (0.69 - 1.0): `-0.0072`

![Loss compare](./loss_compare_20m_tokens.png)
