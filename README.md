# Overview
This is the project for trustworthy retrieval augmented chatbot (TRAQ). The organization of the project is:

- Folder *run*: implementations for TRAQ and other baselines
- Folder *data*: original datasets 
- Folder *collected_data*: collected data including original prompts and generated responses from LLMs
- Folder *collected_results*: evalutation results using different methods, including TRAC, TRAC-P, Bonf, Bonf-P
- Folder *collected_plots*: plots reported in our submission
- Folder *finetuned_models*: weights for finetuned models used in our submission
- File *requirements.txt*: list important packages and corresponding versions

Please download corresponding folders (except *collected results*) from this link: [TRAC](https://drive.google.com/drive/folders/1irO2-Fu-cpaDhOLEnpc7_-bOgdk0zvu3?usp=drive_link); and unzip to the root folder.

# Evaluate different methods
```
cd trac
# evaluate methods using chatgpt
./bash/evaluate_chatgpt.bash
# evaluate methods using llama-2
./bash/evaluate_opensource.bash
# evaluate methods using few-shot prompt 
./bash/evaluate_chatgpt_semantic.bash
# evaluate method using a specific configuration using chatgpt
python trac_chatgpt.py --task $TASK --seed $SEED --alpha $ALPHA
# evaluate method using a specific configuration using llama-2
python trac_opensource.py --task $TASK --seed $SEED --alpha $ALPHA
```

# Analysis evaluation results
```
cd trac/analysis
# analysis results using chatgpt
python analysis.py --task chatgpt
# analysis results using llama-2
python analysis.py --task opensource
```

# collect data
Run jupyter notebooks in folder *trac/collect*
