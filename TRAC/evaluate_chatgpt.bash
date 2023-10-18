#!/bin/bash
for ALPHA in 0.2 0.1
do
    for TASK in nq trivia squad1
    do
        for SEED in 0 21 42
        do
            python evaluate_chatgpt.py \
                --task $TASK \
                --seed $SEED \
                --alpha $ALPHA >> chatgpt.log
        done
    done
done