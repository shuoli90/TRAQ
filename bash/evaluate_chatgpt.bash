#!/bin/bash
# echo -n "" > chatgpt_results.txt
# for ALPHA in 0.5 0.4 0.3 0.2 0.1
# do
#     for TASK in bio nq trivia squad1
#     do
#         for SEED in 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
#         do
#             python trac_chatgpt.py \
#                 --task $TASK \
#                 --seed $SEED \
#                 --alpha $ALPHA
#         done
#     done
# done

for ALPHA in 0.5 0.4 0.3 0.2 0.1
do
    for TASK in bio nq trivia squad1
    do
        for SEED in 15 25 35 45 55 60 65 70 75 80 85 90 95 100
        do
            python trac_chatgpt.py \
                --task $TASK \
                --seed $SEED \
                --alpha $ALPHA
        done
    done
done