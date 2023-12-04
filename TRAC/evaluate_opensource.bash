#!/bin/bash
for ALPHA in 0.5 0.4 0.3 0.2 0.1
do
    for TASK in nq trivia squad1
    do
        for SEED in 10 20 30 40 50
        do
            python evaluate_opensource.py \
                --task $TASK \
                --seed $SEED \
                --alpha $ALPHA
        done
    done
done

# for TASK in nq trivia squad1
# do
#     python evaluate_opensource.py \
#             --task $TASK 
# done