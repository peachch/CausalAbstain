import torch
from ollama import ChatResponse
from ollama import chat
import openai
import os
import time
import numpy as np
import time
import wikipedia as wp
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import math
from scipy.spatial import distance
import ollama

device = "cuda"
def distribution(all_abstains):
    distributions = []
    for abstain_flags in all_abstains:
        counts = {0: 0, 1: 0}
        for value in abstain_flags:
            if value in counts:
                counts[value] += 1

        total_count = sum(counts.values())
        if total_count == 0:
            return [0.0, 0.0]

        probabilities = [counts[0] / total_count, counts[1] / total_count]

        # softmax
        max_prob = max(probabilities)
        exp_probs = [math.exp(p - max_prob) for p in probabilities] 

        sum_exp_probs = sum(exp_probs) 

        softmax_values = [exp_p / sum_exp_probs for exp_p in exp_probs] 
        distributions.append(softmax_values)

    return distributions 

def JSD(distribution1, distribution2):
    jsds = []
    for i in range(len(distribution1)):
        p = np.array(distribution1[i])
        q = np.array(distribution2[i])
        
        # calculate distribution average
        m = 0.5 * (p + q)
        
        # use np.clip to set the scope
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        m = np.clip(m, 1e-10, None)
        
        # JSD
        jsd = 0.5 *(np.sum(p* np.log(p / m)) + np.sum(q * np.log(q / m)))
        jsds.append(jsd)
    return jsds

def llm_init(model_name):
    global device
    global model
    global tokenizer
    global pipeline

    if model_name == "aya_13b":
        device = "cuda"
        # device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-101")
        model = AutoModelForSeq2SeqLM.from_pretrained("CohereForAI/aya-101", device_map="auto")


    if model_name == "gpt3.5" or model_name == "gpt4"or model_name =="claude":
        device = "cuda"
        openai.api_base = "https://openkey.cloud/v1"
        openai.api_key = "sk-anQGO9X0r1zNkkqcCf29D03eEbDd4e37AbF1Ba986863Da53"

def wipe_model():
    global device
    global model
    global tokenizer
    global pipeline
    device = None
    model = None
    tokenizer = None
    pipeline = None
    del device
    del model
    del tokenizer
    del pipeline

# @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, probs = False, temperature = 0.1, max_new_tokens = 200):
    if not model_name == "gpt3.5" and not model_name == "gpt4" and not model_name =="claude"and not model_name =="llama" and not model_name=="deepseek"and not model_name=="phi":

        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, temperature=temperature, do_sample=True)


        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )

        input_length = inputs.input_ids.shape[1]
        if model_name == "aya_13b" or model_name == "aya23_8b":

            generated_tokens = outputs.sequences[:, 1:-1]
        else:
            generated_tokens = outputs.sequences[:, input_length:]
        generated_text = tokenizer.batch_decode(generated_tokens)[0].strip()
        token_probs = {}
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            token_probs[tokenizer.decode(tok).strip()] = np.exp(score.item())

    
    elif model_name == "gpt3.5":

        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    logprobs=True,
                )
        time.sleep(0.1)
        token_probs = {}

        for thing in response['choices'][0]['logprobs']["content"]:
            token_probs[thing["token"]] = np.exp(thing["logprob"])

    elif model_name == "mistral":
        response = chat(model='mistral', messages=[

        {
            'role': 'user',
            'content': prompt,
        },
        ])
        time.sleep(0.1)
        token_probs = {}
        return response['message']['content'].strip()
    elif model_name == "phi":

        response = chat(model='phi4', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        time.sleep(0.1)
        token_probs = {}
        return response['message']['content'].strip()
    elif model_name == "llama":
        # we use ollama to run llama
        response = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        time.sleep(0.1)
        token_probs = {}
        return response['message']['content'].strip()
    elif model_name == "deepseek":
        response = chat(model='deepseek-r1', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
        ])
        time.sleep(0.1)
        token_probs = {}
        return response['message']['content'].strip()
    elif model_name == "claude":

        response = openai.ChatCompletion.create(
                    model="claude-3-7-sonnet-20250219",
                    # model="claude-3-haiku-20240307",
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    logprobs=True,
                )
        time.sleep(0.1)
        token_probs = {}
        if probs:
            return response['choices'][0]['message']['content'].strip(), token_probs
        else:
            return response['choices'][0]['message']['content'].strip()
    
    elif model_name == "gpt4":
        response = openai.ChatCompletion.create(
                    model="gpt-4o-2024-08-06",
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_new_tokens,
                    # logprobs=True,
                )
        time.sleep(0.1)
        token_probs = {}
        if probs:
            return response['choices'][0]['message']['content'].strip(), token_probs
        else:
            return response['choices'][0]['message']['content'].strip()

def answer_parsing(response):
    # mode 1: answer directly after
    temp = response.strip().split(" ")
    for option in ["A", "B", "C", "D", "E"]:
        if option in temp[0]:
            return option
    # mode 2: "The answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the answer is " + option in temp:
            return option.upper()
    # mode 3: "Answer: A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "answer: " + option in temp:
            return option.upper()
    # mode 4: " A/B/C/D/E " or " A/B/C/D/E."
    for option in ["A", "B", "C", "D", "E"]:
        if " " + option + " " in response or " " + option + "." in response:
            return option
    # mode 5: "The correct answer is A/B/C/D/E"
    temp = response.lower()
    for option in ["a", "b", "c", "d", "e"]:
        if "the correct answer is " + option in temp:
            return option.upper()
    # mode 6: "A: " or "B: " or "C: " or "D: " or "E: "
    for option in ["A", "B", "C", "D", "E"]:
        if option + ": " in response:
            return option
    # mode 7: "A/B/C/D/E" and EOS
    try:
        for option in ["A", "B", "C", "D", "E"]:
            if option + "\n" in response or response[-1] == option:
                return option
    except:
        pass
    # mode 8: "true" and "false" instead of "A" and "B" for feedback abstention

    if "true" in response.lower():
        return "A"
    if "false" in response.lower():
        return "B"

    # fail to parse
    print("--------------fail to parse answer ------------------")
    return "Z" # so that its absolutely wrong

prompt = "Question: Who is the 44th president of the United States?\nAnswer:"

