# CausalAbstain Repository
This is the repo for the paper: 

```
conda env create -f causalabstain.yaml
conda activate causal
```
## LLMs
We use ollama to run llama, phi4, etc. 
### download
```
curl -fsSL https://ollama.com/install.sh | sh
```
### run the server and LLM
```
export OLLAMA_HOST="127.0.0.1:8000"
ollama serve
```

```
ollama run llama3.2
```
To use GPT-4o and Claude via API keys, set up your environment or code to include these keys.
```
export OPENAI_API_KEY="YOUR_KEY"
```

### CausalAbstain

```
python causalabstain.py -m llama -d mmlu -s it -l true -r three -n 3 -f True -t test -o 0.5

```
Method parameters:
```
-m MODEL, --model MODEL
                        which language model to use: "llama", "chatgpt", "gpt4", etc.
-d DATASET, --dataset DATASET
                        which dataset in data/: "mmlu", "hellaswag"
-s LANGUAGE, --speak LANGUAGE
                        The native language
-r REALTED, --related RELATED
                        the number of related languages
-n ITER_NUMBER, --iter_number ITERNUMBER
                        the number of iterations
```
Parameters to save file, decide test or evaluate (can be revised accordingly):

```
-l LOCAL, --local LOCAL
                        local copy of preds saved
-f FEEDBACK, -feedback FEEDBACK
                        Whether to save generated feedback to the local file
-t TEST_OR_EVALUATION, -test TEST_OR_EVALUATION
                        save file for different location
-o PORTION, --portion PORTION
                        portion of the dataset to use, 0-1
```



### Models & Metrics
lm_utils.py provides inference code for LLMs, and the method of calculating the score via Jensen-Shannon Divergence (JSD) in Section 3.2 of our paper. You can add the models if need. metrics.py provides the implementation of AbstainQA metrics.

### Baselines
The baseline comparisons in our paper are detailed in the baselines/ file. Please refer to baselines/baselines.md for the relevant information.
