import json
import argparse
import numpy as np
from pathlib import Path

def aggregate_results(model_type, tagging_scheme, num_runs, output_dir):
    all_metrics = []
    for i in range(1, num_runs + 1):
        report_file = Path(output_dir) / f"{model_type}_{tagging_scheme}_run{i}_report.json"
        with open(report_file, 'r') as f:
            report = json.load(f)
            all_metrics.append(report['test_metrics'])
    
    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        values = [run[metric] for run in all_metrics]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Save aggregated results
    aggregated_report = {
        'model_type': model_type,
        'tagging_scheme': tagging_scheme,
        'num_runs': num_runs,
        'aggregated_metrics': avg_metrics
    }
    
    output_file = Path(output_dir) / f"{model_type}_{tagging_scheme}_aggregated_report.json"
    with open(output_file, 'w') as f:
        json.dump(aggregated_report, f, indent=2)
    
    print(f"Aggregated report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--tagging_scheme', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='../results')
    args = parser.parse_args()
    
    aggregate_results(args.model_type, args.tagging_scheme, args.num_runs, args.output_dir)