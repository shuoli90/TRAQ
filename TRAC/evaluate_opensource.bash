#!/bin/bash
echo -n "" > opensource_results.txt
for ALPHA in 0.5 0.4 0.3 0.2 0.1
do
    for TASK in nq trivia squad1
    do
        for SEED in 0 21 42
        do
            python evaluate_opensource.py \
                --task $TASK \
                --seed $SEED \
                --alpha $ALPHA
        done
    done
done