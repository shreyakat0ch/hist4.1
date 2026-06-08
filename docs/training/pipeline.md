# Training Pipeline

The Deep-H training pipeline is optimized for stability and performance on highly imbalanced multi-task epigenomic datasets.

## Performance Optimizations

Deep-H leverages several PyTorch features to maximize GPU utilization and training speed:

<div class="feature-grid">
  <div class="feature-card coral">
    <h3>Automatic Mixed Precision (AMP)</h3>
    <p>Forward passes run in `float16` to halve memory usage and double throughput on Tensor Cores, while gradients are scaled to prevent underflow.</p>
  </div>
  <div class="feature-card turquoise">
    <h3>Gradient Checkpointing</h3>
    <p>Mamba layers use PyTorch's `checkpoint()` to trade minor compute overhead for massive memory savings, allowing longer sequences.</p>
  </div>
  <div class="feature-card lavender">
    <h3>Gradient Accumulation</h3>
    <p>Physical batch size is 256 (memory limit), but gradients are accumulated over 4 steps for an <strong>effective batch size of 1024</strong>, stabilizing the multi-task loss.</p>
  </div>
  <div class="feature-card golden">
    <h3>Persistent Workers</h3>
    <p>The DataLoader uses 16 workers with `persistent_workers=True` and `file_system` sharing strategy to eliminate epoch-to-epoch startup latency.</p>
  </div>
</div>

## Learning Rate Schedule

The optimizer uses a **Warmup-Cosine** schedule (`SequentialLR`):

1. **Warmup Phase:** Linearly increases from 1e-6 to 3e-4 over the first 3,000 steps. This prevents early divergence when the network is randomly initialized.
2. **Cosine Decay Phase:** Gradually decays the learning rate following a cosine curve for the remainder of the 50 epochs, allowing fine-grained convergence at the end of training.

```python
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)

warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-3, total_iters=3000
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_opt_steps - 3000
)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[3000]
)
```

## Checkpointing & Early Stopping

### Automatic Resume
The script constantly maintains a `latest_checkpoint.pt`. If training is interrupted (e.g., cluster node preemption), simply restarting `train.py` will automatically load the checkpoint and resume from the exact epoch and scheduler step.

### Early Stopping
Training stops if the **validation mean Pearson correlation** does not improve for 10 consecutive epochs (`EARLY_STOPPING_PATIENCE = 10`).

### Best Model Selection
The model that achieves the highest mean Pearson correlation across all 4 marks on the validation set is saved as `best_model.pt`.

## Multi-Processing Workaround

!!! warning "DataLoader Zombie Threads"
    A known issue in PyTorch combined with complex C++ extensions (like `mamba-ssm`) can cause `LLVM ERROR: pthread_join failed` crashes at the end of epochs.
    
    Deep-H implements a dual-strategy workaround:
    1. `torch.multiprocessing.set_sharing_strategy('file_system')`
    2. The Validation DataLoader uses `num_workers=0` (runs in the main process) to prevent worker-process tearing during evaluation phases.

## Metrics Tracking

During training, Deep-H tracks and logs an extensive suite of metrics to `runs/training_log.csv`:

**Scalar Head Metrics:**
- Loss (Huber + Pearson)
- Pearson Correlation (Overall & Per-Mark)
- AUPRC (Area Under Precision-Recall Curve)
- MCC (Matthews Correlation Coefficient)

**Track Head Metrics:**
- Track BCE Loss
- Track F1 Score

Visualizations of peak predictions vs ground truth are automatically generated for the validation set every epoch and saved to `runs/plots/`.

---

**Next:** [Loss Functions →](losses.md)
