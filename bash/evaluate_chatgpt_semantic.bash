#!/bin/bash
# echo -n "" > chatgpt_results.txt
for TASK in squad1
do
    for SEED in 10 20 30 40 50
    do
        for alpha in 0.5 0.4 0.3 0.2 0.1
        do
            python trac_chatgpt_semantic.py \
                --task $TASK \
                --seed $SEED \
                --alpha $alpha
        done
    done
done