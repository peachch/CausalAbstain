import json
import argparse
import lm_utils
import metrics
import random
from tqdm import tqdm
device = "cuda"

if __name__ == "__main__":

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="which language model to use") # "aya_13b", "chatgpt", "gpt4", llama, phi 
    argParser.add_argument("-d", "--dataset", help="which dataset") # "mmlu", "hellaswag"
    argParser.add_argument("-s", "--speak", help="speak which language") # "nl", "es", etc.
    argParser.add_argument("-o", "--portion", default = 1.0, help="portion of the dataset to use")
    argParser.add_argument("-l", "--local", default = False, help="local copy of preds saved")
    argParser.add_argument("-f", "--feedback", default = False, help ="whether to save generated feedbacks in a seperate file")
    argParser.add_argument("-r", "--related", default = "single", help ="voted method when consider related languages")
    argParser.add_argument("-n", "--iter_number", type=int, default = 3, help ="iterated number of obtain responses")
    argParser.add_argument("-t", "--test_or_evaluation",  default = "test", help ="classify save files")

    args = argParser.parse_args()
    model_name = args.model
    dataset = args.dataset
    speak = args.speak
    approach_type = "self"
    portion = args.portion
    local_flag = args.local
    feedback_flag = args.feedback

    lm_utils.llm_init(model_name)


    language_list = ["English", "Russian", "German", "Chinese", "French", "Spanish", "Italian", "Dutch", "Vietnamese",
                     "Indonesian", "Arabic", "Hungarian", "Romanian", "Danish", "Slovak", "Ukrainian", "Catalan", "Serbian", "Croatian", "Hindi",
                     "Bengali", "Tamil", "Nepali", "Malayalam", "Marathi", "Telugu", "Kannada", "Deutsch"]
    
    # revise the language according to your need
    language_related_dict = {
        "zh": ["Chinese", "Chinese", "Chinese", "English", "Russian", "German", "Italian", "Dutch", "Arabic"],
        "id": ["English","Catalan", "Russian", "Indonesian", "German"],
        "ar": ["Chinese", "Italian", "Dutch", "Arbic", "English"],
        "bn": ["Arabic", "Hindi", "Bengali", "Nepali", "Vietanamese", "English", "Telugu", "kannada", "Russian"],
        "ta": ["Arabic", "Hindi", "Bengali","Chinese", "Italian", "Dutch","Malayalam","Marathi", "Telugu"],
        "ne": ["Kanaada", "Telugu", "Hindi","Nepali","English"],
        "te": ["Kannada", "Russian", "Catalan","Telugu","English"],
        "kn": ["Telugu", "Malaayalam", "Tamil","Kannada","English"],
        "it": ["Catalan", "Romanian", "Ukrainian","Italian","English"]
    }
    
    with open("data/" + dataset + "/" + dataset + "_" + speak + ".json", "r") as f:

        data = json.load(f)
        data["dev"] = data["dev"][:int(len(data["dev"])*float(portion))]
        data["test"] = data["test"][:int(len(data["test"])*float(portion))]

        def dict_to_list(data_dict):
            list_length = len(next(iter(data_dict.values())))
            result_list = [[] for _ in range(list_length)]
            
            # iterate
            for key in sorted(data_dict.keys()):
                for i in range(list_length):
                    result_list[i].append(data_dict[key][i])
            
            return result_list
        # obtain correct flags
        N = args.iter_number 
        n_times_abstain_x = {}
        n_times_abstain_source = {}
        n_times_abstain_english = {}
        answers = []
        correct_flags = []
        print("--------get direct answer----------")
        for d in tqdm(data["test"]):
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. The answer:"
            response = lm_utils.llm_response(original_prompt, model_name, probs=False, max_new_tokens=5)
            print(lm_utils.answer_parsing(response))
            if lm_utils.answer_parsing(response) == d["answer"]:
                correct_flags.append(1)
            else:
                correct_flags.append(0)
            # Question -> Answer
            answers.append(response)
        print("--------get no fb abstain or not [nde]----------")
        while N >=1 :    
            feedback_1 = []
            feedback_2 = []
            feedback_3 = []
            abstain_flags = []
            abstain_flags_fb1 = []
            abstain_flags_x = []
            abstain_scores = []
            abstain_scores_fb1 = []

            for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
                original_prompt = "Question: " + d["question"] + "\n"
                for key in d["choices"].keys():
                    original_prompt += (key + ": " + d["choices"][key] + "\n")
                original_prompt += "Choose one answer from the above choices. Proposed answer:"

                response = original_prompt + " " + answers[i].strip() + "\nIs the proposed answer True or False?"
                response = lm_utils.llm_response(response, model_name, probs=False, max_new_tokens=5)

                parse_answer = lm_utils.answer_parsing(response)
                if parse_answer == "A":
                    abstain_flags_x.append(0)
                elif parse_answer == "B":
                    abstain_flags_x.append(1)
                else:
                    # print("Error: abstain flag not found")
                    abstain_flags_x.append(random.randint(0, 1))

            n_times_abstain_x[N] = abstain_flags_x       
            n_times_abstain_english[N] = abstain_flags
            N -= 1 
    # print(n_times_abstain_x)

    # get the score of NDE
    N = args.iter_number
    n_times_lists_x = dict_to_list(n_times_abstain_x)
    y_base = [[1/2 for _ in range(2)] for _ in range(len(n_times_lists_x))]    
    x_distribution = lm_utils.distribution(n_times_lists_x)
    ndes = lm_utils.JSD(x_distribution, y_base)
    
    # causal judge
    from collections import Counter
    causal_abstain_flags = []
    def most_frequesnt_element(input):
        frequency_count = Counter(input)
        most_common_element, most_common_count = frequency_count.most_common(2)[0]
        try:
            most_common_element_2, most_common_count_2 = frequency_count.most_common(2)[1]
        except:
            # when all the element is the same, directly output the first order.
            return most_common_element, most_common_count, True
        if most_common_count != most_common_count_2:
            return most_common_element, most_common_count, True
        elif most_common_count == most_common_count_2:
            return most_common_element, most_common_count, False 
    assert len(ndes) == len(data["test"])
 
    tie_or_nde_flags = []
    tie_or_nde = []
    tie_nde_list = []
    related_feedbacks = []
    tie_related_maps = []
    
    print("--------get feedback and review with feedback [TIE]----------")
    for d, i in tqdm(zip(data["test"], range(len(data["test"])))):
        if args.related == "three":
            def calculate_related_language(d, prompt_feedback_related, N, answer):
                prompt_feedback_experts = prompt_feedback_related

                abstain_flags_dict = {}
                abstain_scores_dict = {}
                related_feedback = []
                while N>= 1:
                    abstain_flags_fb = []
                    abstain_scores_fb = []
                    feedback_related_1 = []
                    response = lm_utils.llm_response(prompt_feedback_experts, model_name, probs=False, temperature=1)
                    response = response.split("\n")[0].strip()
                    if len(response) == 0: # to avoid no generated feedback in multilingual settings
                        response = "No feedback provided."
                    feedback_related_1 = response
                    prompt_area_chair_fb = "Question: " + d["question"] + "\n"

                    for key in d["choices"].keys():
                        prompt_area_chair_fb += (key + ": " + d["choices"][key] + "\n")
                    related_feedback.append(feedback_related_1)

                    prompt_area_chair_fb += "Choose one answer from the above choices. Proposed answer: " + answer.strip() + "\n\nFeedback: " + feedback_related_1.strip() + "\n\nBased on the feedback, is the proposed answer True or False? Please respond clearly with 'True' or 'False'."
                        
                    response = lm_utils.llm_response(prompt_area_chair_fb, model_name, probs=False, max_new_tokens=10)

                    if lm_utils.answer_parsing(response) == "A":
                        abstain_flags_fb.append(0)
                    elif lm_utils.answer_parsing(response) == "B":
                        abstain_flags_fb.append(1)
                    else:
                        # print("Error: abstain flag not found")
                        abstain_flags_fb.append(random.randint(0, 1))

                    N -= 1
                    abstain_flags_dict[N] = abstain_flags_fb
                    abstain_scores_dict[N] = abstain_scores_fb
                return abstain_flags_dict, abstain_scores_dict, related_feedback
            
            original_prompt = "Question: " + d["question"] + "\n"
            for key in d["choices"].keys():
                original_prompt += (key + ": " + d["choices"][key] + "\n")
            original_prompt += "Choose one answer from the above choices. Proposed answer:"

            prompt_feedback = original_prompt + " " + answers[i].strip() + "\nPlease review the proposed answer and provide a paragraph of feedback on its correctness. Feedback should be in <language>.\nFeedback:"
            prompt_feedback_related = []
            for k in range(3):
                specified_language = language_related_dict[speak][k]
                prompt_feedback_related.append(prompt_feedback.replace("<language>", specified_language))
            assert len(prompt_feedback_related) == 3 
            try:
                abstain_related_flag_1, abstain_related_score_1, related_feedback_1 = calculate_related_language(d, prompt_feedback_related[0], N, answers[i])
                abstain_related_flag_2, abstain_related_score_2, related_feedback_2 = calculate_related_language(d, prompt_feedback_related[1], N, answers[i])
                abstain_related_flag_3, abstain_related_score_3, related_feedback_3 = calculate_related_language(d, prompt_feedback_related[2], N, answers[i])
                related_feedbacks.append([related_feedback_1, related_feedback_2, related_feedback_3])

                # calculate new TIE score
                abstain_related_1 = dict_to_list(abstain_related_flag_1)
                related1_feedback_n_distributions = lm_utils.distribution(abstain_related_1)
                ties_related1 = lm_utils.JSD(related1_feedback_n_distributions, [x_distribution[i]])
                
                abstain_related_2 = dict_to_list(abstain_related_flag_2)
                related2_feedback_n_distributions = lm_utils.distribution(abstain_related_2)
                ties_related2 = lm_utils.JSD(related2_feedback_n_distributions, [x_distribution[i]])
                
                abstain_related_3 = dict_to_list(abstain_related_flag_3)
                related3_feedback_n_distributions = lm_utils.distribution(abstain_related_3)
                ties_related3 = lm_utils.JSD(related3_feedback_n_distributions, [x_distribution[i]])
                
                tie_related_abstain_map = {
                            'tie_related1': [abstain_related_1[0], ties_related1[0]],
                            'tie_related2': [abstain_related_2[0], ties_related2[0]],
                            'tie_related3': [abstain_related_3[0], ties_related3[0]],
                        }

                # record the process
                this_tie_or_nde_flags_string = ""
                join_voted = []

                if ndes[i] > ties_related1[0] and ndes[i] > ties_related2[0] and ndes[i] > ties_related3[0]:
                    join_voted = []
                else:
                    join_voted = abstain_related_1[0] + abstain_related_2[0] + abstain_related_3[0]
                if join_voted == []:
                    causal_abstain_flags.append(most_frequesnt_element(n_times_lists_x[i])[0])
                    this_tie_or_nde_flags_string += "ndes"
                else:
                    this_tie_or_nde_flags_string += "related"
                    majorty_voted, count, flag = most_frequesnt_element(join_voted)
                    if flag:
                        causal_abstain_flags.append(majorty_voted)
                    else:
                        rdm = random.randint(0,1)
                        causal_abstain_flags.append(rdm)
                        this_tie_or_nde_flags_string += "& random"
                tie_or_nde_flags.append(this_tie_or_nde_flags_string)
                tie_related_maps.append(tie_related_abstain_map) 
            except Exception as e:
                print("sth wrong, pass the error, record nothing")
                print(e)
                related_feedbacks.append([])
                causal_abstain_flags.append(1)
                this_tie_or_nde_flags_string = "error"
                tie_related_abstain_map = {
                            'tie_related1': [],
                            'tie_related2': [],
                            'tie_related3': [],
                        }
                this_tie_or_nde_flags_string = ""
                tie_or_nde_flags.append(this_tie_or_nde_flags_string)
                tie_related_maps.append(tie_related_abstain_map) 
    # the causal abstain decisions
    print(causal_abstain_flags)

    causal_abstain_scores = None
    if local_flag:
        # record in differnent file, can remove/revise accordingly
        if args.test_or_evaluation == "test":
            with open("preds_test/" + args.related + str(N) + model_name + "_" + dataset + "_" + speak + "_causal_details.json", "w") as f:
                json.dump({"ndes":ndes,"n_times_abstain_x":n_times_lists_x, "correct_flags": correct_flags, "causal_abstain_flags": causal_abstain_flags, "causal_abstain_scores": causal_abstain_scores, "tie_ndes_list":tie_nde_list, "tie_or_nde":tie_or_nde, "tie_or_nde_flags":tie_or_nde_flags, "related_fb": related_feedbacks, "tie_related_maps": tie_related_maps}, f, indent=4)
        else:
            with open("preds/" +args.related + str(N) + model_name + "_" + dataset + "_" + speak + "_causal_details.json", "w") as f:
                json.dump({"ndes":ndes,"n_times_abstain_x":n_times_lists_x, "correct_flags": correct_flags, "causal_abstain_flags": causal_abstain_flags, "causal_abstain_scores": causal_abstain_scores, "tie_ndes_list":tie_nde_list, "tie_or_nde":tie_or_nde, "tie_or_nde_flags":tie_or_nde_flags, "related_fb": related_feedbacks, "tie_related_maps": tie_related_maps}, f, indent=4)
        
    print("------------------")
    print("Approach: causal")
    print("Model:", model_name)
    print("Dataset:", dataset)
    print("Language:", speak)
    print("Type:", approach_type)
    print(metrics.compute_metrics(correct_flags, causal_abstain_flags, causal_abstain_scores))
    print("------------------")
