from nats_bench import create
import pandas as pd
from tqdm import tqdm

nats_path = "_"

api = create(
    nats_path,
    'tss',
    fast_mode=False,
    verbose=True
)

results = []

for arch_id in tqdm(range(len(api))):
    # Get full info for each arch
    info = api.get_more_info(arch_id, dataset='cifar10', hp='200')

    results.append({
        "arch_id": arch_id,
        "train_loss": info.get("train-loss"),
        "train_accuracy": info.get("train-accuracy"),
        "train_per_time": info.get("train-per-time"),
        "train_all_time": info.get("train-all-time"),
        "test_loss": info.get("test-loss"),
        "test_accuracy": info.get("test-accuracy"),
        "test_per_time": info.get("test-per-time"),
        "test_all_time": info.get("test-all-time"),
        "comment": info.get("comment", "")
    })

# Convert to csv
df = pd.DataFrame(results)
df.to_csv("cifar10_tss_full_metrics.csv", index=False)

print("Export complete. CSV saved as 'cifar10_tss_full_metrics.csv'.")