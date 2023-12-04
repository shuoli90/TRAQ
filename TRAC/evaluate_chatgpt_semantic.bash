#!/bin/bash
# echo -n "" > chatgpt_results.txt
for TASK in squad1
do
    python evaluate_chatgpt_semantic.py \
        --task $TASK 
done