import os
import glob
import pandas as pd
import numpy as np

def parse_logs():
    csv_files = glob.glob("**/experiment_log_*.csv", recursive=True)
    if not csv_files:
        print("No experiment log CSV files found.")
        return

    data = []
    for f in csv_files:
        if os.path.dirname(f) == "":
            continue
        filename = os.path.basename(f)
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
                
            # Determine method type and configurations from filename or columns
            parts = filename.replace(".csv", "").split("_")
            
            # Common parse
            dataset = "qiang" if "qiang" in filename else "noble"
            n_shot = 3 if "3shot" in filename else (5 if "5shot" in filename else None)
            
            if "baseline" in filename:
                if "resnet34" in filename:
                    method = "ResNet-34 Baseline"
                elif "swin3d" in filename:
                    if "da_baseline" in filename:
                        loss_type = "mmd" if "mmd" in filename else "coral"
                        method = f"Swin3D + {loss_type.upper()} (DA Feature-Only)"
                    else:
                        method = "Swin3D Baseline"
                
                # Ignore/drop lambda_residual for baselines as it is just a parser/naming artifact
                if "lambda_residual" in df.columns:
                    df = df.drop(columns=["lambda_residual"])
            elif "joint" in filename:
                loss_type = "mmd" if "mmd" in filename else "coral"
                if "ablation" in filename:
                    # extract the suffix following 'ablation_': e.g., 'stn', 'intensity', etc.
                    ablation_part = filename.split("ablation_")[-1].replace(".csv", "")
                    method = f"Swin3D + {loss_type.upper()} (Ablation: {ablation_part.upper()}-only)"
                else:
                    method = f"Swin3D + {loss_type.upper()} (Proposed)"
            else:
                continue

            # Group by parameters to compute mean/std across seeds (usually epoch 30 is the last row)
            # Find the final epoch for each seed
            last_epochs = df.groupby(["seed", "n_shot"]).last().reset_index()
            
            if "joint" in filename:
                # Joint runs have lambda_residual. Find the best lambda_residual by mean acc_R
                lambda_groups = df.groupby(["lambda_residual", "seed"]).last().reset_index()
                best_lambda = None
                best_mean = -1
                best_std = 0
                
                for lam, group in lambda_groups.groupby("lambda_residual"):
                    mean = group["acc_R"].mean()
                    std = group["acc_R"].std()
                    if mean > best_mean:
                        best_mean = mean
                        best_std = std
                        best_lambda = lam
                
                data.append({
                    "Dataset": dataset.capitalize(),
                    "Shots": f"{n_shot}-shot",
                    "Method": method,
                    "Accuracy": f"{best_mean:.2f}% ± {best_std:.2f}%",
                    "Best Lambda": f"{best_lambda:.2f}",
                    "sort_key": (dataset, n_shot, method)
                })
            else:
                mean = last_epochs["acc_R"].mean()
                std = last_epochs["acc_R"].std()
                data.append({
                    "Dataset": dataset.capitalize(),
                    "Shots": f"{n_shot}-shot",
                    "Method": method,
                    "Accuracy": f"{mean:.2f}% ± {std:.2f}%",
                    "Best Lambda": "N/A",
                    "sort_key": (dataset, n_shot, method)
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not data:
        print("No valid statistics could be computed from the logs.")
        return

    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values(by="sort_key").drop(columns=["sort_key"])
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    try:
        print(results_df.to_markdown(index=False))
    except ImportError:
        print(results_df.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    parse_logs()
