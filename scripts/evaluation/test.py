import numpy as np
import json
from collections import defaultdict
import argparse
import re
import math
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def calculate_similarity_score(predicted_str, ground_truth_str):

    predicted_clean = predicted_str.strip().lower()
    ground_truth_clean = ground_truth_str.strip().lower()
    
    if predicted_clean == ground_truth_clean:
        return 1.0
    

    if ground_truth_clean in predicted_clean or predicted_clean in ground_truth_clean:
        shorter = min(len(predicted_clean), len(ground_truth_clean))
        longer = max(len(predicted_clean), len(ground_truth_clean))
        return shorter / longer
    
    import re
    if re.search(r'[\u4e00-\u9fff]', predicted_clean) or re.search(r'[\u4e00-\u9fff]', ground_truth_clean):
        pred_chars = set(predicted_clean)
        gt_chars = set(ground_truth_clean)
        common_chars = pred_chars.intersection(gt_chars)
        
        if len(pred_chars) == 0 or len(gt_chars) == 0:
            return 0
        
        precision = len(common_chars) / len(pred_chars)
        recall = len(common_chars) / len(gt_chars)
        
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    return calculate_f1_score(predicted_str, ground_truth_str)


def evaluate(args):


    def read_jsonl(path):
        data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    try:
        data = read_jsonl(args.data_path)
    except:
        data = load_json(args.data_path)



    # processed subsplits file to category
    # with open(f"android_control_test_subsplits.json", "r") as file:
    #     test_subsplits = json.load(file)

    # ======================================================================== #
    #                          Results on Low-level
    # ======================================================================== #
    mis_click_wait_num = 0
    step_acc_res_dict = defaultdict(int)
    sample_number_dict = defaultdict(int)
    # for pred, gt in zip(prediction, ground_truth):
    for x in data:
        pred = x['response']
        gt = x['label']


        try:
            pred_action = pred.split("actions:\n")[1].strip("<|im_end|>").strip()  # <im_end> for qwen2-vl
        except:
            try:
                pred_action = pred.split("Actions:")[1].strip("<|im_end|>").strip()
            except:
                pred_action = "invalid action"

        gt_action = gt.split("Actions:\n")[1].strip()


        sample_number_dict["full"] += 1

        sample_number_dict[gt_action.split()[0]] += 1

        gt_action_type = gt_action.split()[0]
        # if gt_action_type == 'HOTKEY':
        #     print(gt_action)
        
        pred_action_parts = pred_action.split()
        if len(pred_action_parts) == 0:
            pred_action_type = "invalid_action"
        else:
            pred_action_type = pred_action_parts[0]
        # print(pred_action_type)

        # calculate step acc based on types
        if gt_action_type == pred_action_type:
            step_acc_res_dict["type_match"] += 1
            step_acc_res_dict[gt_action_type + "_type_match"] += 1
            if gt_action_type in ["CLICK", "LONG_PRESS", "MOVETO", "RIGHT_CLICK", "DOUBLE_CLICK"]:  # evaluate click type
                step_acc_res_dict["click_match"] += 1
                try: 
                    pred_coords = re.findall(r'\d+\.?\d*', pred_action)
                    pred_x, pred_y = float(pred_coords[0]), float(pred_coords[1])
                except:
                    pred_x, pred_y = -100, -100
                gt_coords = re.findall(r'\d+\.?\d*', gt_action)
                gt_x, gt_y = float(gt_coords[0]), float(gt_coords[1])

                if math.sqrt((pred_x - gt_x) ** 2 + (
                        pred_y - gt_y) ** 2) <= 0.14 * 1000:  

                    step_acc_res_dict["full"] += 1
                    step_acc_res_dict[gt_action_type + "_all_match"] += 1

            elif gt_action_type == "OPEN_APP":
                try: 
                    gt_content = gt_action.split('[')[1].split(']')[0].strip()
                    pred_content = pred_action.split('[')[1].split(']')[0].strip()
                    similarity_score = calculate_similarity_score(pred_content, gt_content)
                    
                    if gt_action == pred_action or similarity_score > 0.5:
                        step_acc_res_dict["full"] += 1
                        step_acc_res_dict[gt_action_type + "_all_match"] += 1
                except:
                    pass  

            elif gt_action_type == "TYPE":
                try:
                    gt_content = gt_action.split('[')[1].split(']')[0].strip()
                    pred_content = pred_action.split('[')[1].split(']')[0].strip()
                    similarity_score = calculate_similarity_score(pred_content, gt_content)
                    
                    if pred_action == gt_action or similarity_score > 0.5:
                        step_acc_res_dict["full"] += 1
                        step_acc_res_dict[gt_action_type + "_all_match"] += 1
                except:
                    pass  

            elif gt_action_type == "COPY":
                try:
                    gt_content = gt_action.split('[')[1].split(']')[0].strip()
                    pred_content = pred_action.split('[')[1].split(']')[0].strip()
                    similarity_score = calculate_similarity_score(pred_content, gt_content)
                    
                    if gt_action == pred_action or similarity_score > 0.5:
                        step_acc_res_dict["full"] += 1
                        step_acc_res_dict[gt_action_type + "_all_match"] += 1
                except:
                    pass 

            elif gt_action == pred_action:  # evaluate other types
                step_acc_res_dict["full"] += 1
                step_acc_res_dict[gt_action_type + "_all_match"] += 1

    # Print the low-level results
    logger.info("=" * 30 + " AC Step Acc " + f"{args.split} " + "=" * 30)

    logger.info(f"type_match acc: %f" % (step_acc_res_dict[f"type_match"] / sample_number_dict[f"full"]))
    logger.info(f"grounding acc: %f" % (step_acc_res_dict[f"CLICK_all_match"] / step_acc_res_dict[f"CLICK_type_match"]))
    logger.info("Acc: %f" % (step_acc_res_dict["full"] / sample_number_dict["full"]))

    import csv

    type_match_acc = 100*step_acc_res_dict["type_match"] / sample_number_dict["full"]
    grounding_acc = 100*step_acc_res_dict["CLICK_all_match"] / step_acc_res_dict["CLICK_type_match"]
    full_acc = 100*step_acc_res_dict["full"] / sample_number_dict["full"]

    row = {
        "path": os.path.basename(args.data_path),
        "type_match_acc": type_match_acc,
        "grounding_acc": grounding_acc,
        "full_acc": full_acc,
        "sample_count": sample_number_dict["full"]
    }

    if args.output_suffix:
        csv_file = f"results_{args.output_suffix}.csv"
    else:
        csv_file = "results.csv"
    
    write_header = False
    try:
        with open(csv_file, 'r', newline='') as f:
            pass
    except FileNotFoundError:
        write_header = True

    import fcntl
    with open(csv_file, 'a', newline='') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)  
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  

    # logger.info("iid: %f" % (step_acc_res_dict["IDD"] / sample_number_dict["IDD"]))
    # logger.info("app_unseen: %f" % (step_acc_res_dict["app_unseen"] / sample_number_dict["app_unseen"]))
    # logger.info("task_unseen: %f" % (step_acc_res_dict["task_unseen"] / sample_number_dict["task_unseen"]))
    # logger.info("category_unseen: %f" % (step_acc_res_dict["category_unseen"] / sample_number_dict["category_unseen"]))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file_path', type=str, default='<prediction_file.jsonl>')
    parser.add_argument('--ground_truth_file_path', type=str, default='<ground_truth_file.jsonl>')
    parser.add_argument('--model_id', type=str, default="")
    parser.add_argument('--split', type=str, default="low")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--output_path', type=str, default='results/score/')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output CSV file to avoid conflicts")
        # parser.add_argument("--save_failed_generation", action='store_true', default=False)




    args = parser.parse_args()

    # test_main(args.data_path)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    file_handler = logging.FileHandler(args.output_path + f"score.log", mode="w")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    evaluate(args)
