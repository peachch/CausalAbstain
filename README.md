# CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy
<p align="center">
  <a href="https://arxiv.org/abs/2506.00519">
    <img src="https://img.shields.io/badge/arXiv-arxiv:2506.00519-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://komarev.com/ghpvc/?username=peachch&label=Page%20Views&color=00FF00" alt="Page Views">
  <img src="https://img.shields.io/github/stars/peachch/CausalAbstain?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/peachch/CausalAbstain?style=social" alt="GitHub forks">
</p>

<p align="center">
  <img src="https://github.com/peachch/CausalAbstain/blob/master/imgs/multilingual_combine%20(1).png" alt="Logo" width="300"/>
</p>

#### This repository is for CausalAbstain. For more details, please refer to our [paper](https://arxiv.org/abs/2506.00519).

## Abstract
Large Language Models (LLMs) often exhibit  knowledge disparities across languages. Encouraging LLMs to abstain when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce CausalAbstain, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that CausalAbstain effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (CASUALNATIVE) and multilingual (CAUSAL-MULTI) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks.  Our code and data are open-sourced at https://github.com/peachch/CausalAbstain.

## Getting Started
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
python causalabstain.py -m gpt3.5 -d mmlu -s zh -l true -r three -n 3 -f True -t test -o 0.5
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
lm_utils.py provides inference code for LLMs, and the method of calculating the score via Jensen-Shannon Divergence (JSD) in Section 3.2 of our paper. You can add the models if needed. metrics.py provides the implementation of AbstainQA metrics.

### Baselines
The baseline comparisons in our paper are detailed in the baselines/ file. Please refer to baselines/baselines.md for the relevant information.


### Citation
If you find our work helpful, please kindly cite our paper:

```
@article{sun2025causalabstain,
  title={CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention},
  author={Sun, Yuxi and Zuo, Aoqi and Gao, Wei and Ma, Jing},
  journal={arXiv preprint arXiv:2506.00519},
  year={2025}
}
```
