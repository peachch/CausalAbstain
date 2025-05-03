
### The baselines
We follow some of the baselinse from https://github.com/BunsenFeng/M-AbstainQA.

The baselines are named: approach-<name>.py file, which contains the implementation of the corresponding approach. Shared parameters for each approach :
```
-m MODEL, --model MODEL
                        which language model to use: "llama", "gpt3.5", "gpt4" etc.
-d DATASET, --dataset DATASET
                        which dataset in data/: "mmlu", "hellaswag"
-o PORTION, --portion PORTION
                        portion of the dataset to use, 0-1
-l LOCAL, --local LOCAL
                        local copy of preds saved
```
An example:
```
python approach-conflict.py -m llama -d mmlu -s kn -l True -o 0.5
```
