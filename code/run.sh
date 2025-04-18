#!/bin/bash

# List of models to run
models=("dfm" "distilbert" "xlmroberta" "longformer" "eurobert" "gina")
tagging_schemes=("joint" "separate" "metaphor" "hyperbole")
num_runs=3

for model in "${models[@]}"; do
    for scheme in "${tagging_schemes[@]}"; do
        echo "Running experiments for $model with $scheme tagging scheme"
        for i in $(seq 1 $num_runs); do
            python train_dafig.py --model_type $model --tagging_scheme $scheme --run_number $i
        done
        
        # Aggregate results
        python aggregate_results.py --model_type $model --tagging_scheme $scheme --num_runs $num_runs
    done
done