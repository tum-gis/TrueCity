#!/bin/bash

# Set the list of real_ratio values
ratios=(0 25 50 75 100)

# Allow override via env var DATA_PATH; default to repo-relative EARLy dir
DATA_PATH=${DATA_PATH:-"../EARLy/datav2_final"}

# Loop through each ratio and run the training
for ratio in "${ratios[@]}"
do
    echo "===================================="
    echo "Starting training with real_ratio=$ratio"
    echo "===================================="

    # Run the training
    python train.py --real_ratio=$ratio

    # Optional: wait 10 seconds between runs
    echo "Finished training with real_ratio=$ratio"
    echo "Sleeping for 10 seconds before next run..."
    sleep 10
done

echo "===================================="
echo "All trainings completed."
echo "===================================="
