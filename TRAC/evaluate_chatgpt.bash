#!/bin/bash
# echo -n "" > chatgpt_results.txt
for ALPHA in 0.5 0.4 0.3 0.2 0.1
do
    for TASK in bio nq trivia squad1
    do
        for SEED in 10 20 30 40 50
        do
            python evaluate_chatgpt.py \
                --task $TASK \
                --seed $SEED \
                --alpha $ALPHA
        done
    done
done

# for ALPHA in 0.5 0.4 0.3 0.2 0.1
# do
#     for SEED in 10 20 30 40 50
#     do
#         python evaluate_chatgpt.py \
#             --seed $SEED \
#             --alpha $ALPHA \
#             --fewshot
#     done
# done

for TASK in bio nq trivia squad1
do
    python evaluate_chatgpt.py \
            --task $TASK 
done

# for ALPHA in 0.1
# do
#     for TASK in nq trivia squad1
#     do
#         for SEED in 0 21 42
#         do
#             python evaluate_chatgpt.py \
#                 --task $TASK \
#                 --seed $SEED \
#                 --alpha $ALPHA
#         done
#     done
# done

# for ALPHA in 0.1
# do
#     for SEED in 0 21 42
#     do
#         python evaluate_chatgpt.py \
#             --seed $SEED \
#             --alpha $ALPHA \
#             --fewshot
#     done
# done