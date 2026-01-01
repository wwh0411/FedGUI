#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-dataset JSONL Conversion Tool
Function: Convert train.jsonl format to target.jsonl, assign clients by episode, and split by steps.
"""

import json
import random
import os
from typing import List, Dict, Any

# =================================================================
# 🔧 Configuration Area
# =================================================================

# Distribution mode: "iid", "full-random", "iid-random", "iid-level", 
# "skew-1", "skew-1-random", "skew-device", "skew-device-random", "skew-dataset"
DISTRIBUTION_MODE = "iid"  
IID_LEVEL_DISTRIBUTION = [15, 25, 35, 45, 60, 80, 100, 110, 130]  # Total = 600
NUM_CLIENTS = 9  

# Output File Configuration
OUTPUT_FILE = "./output/converted_data.jsonl"

# Dataset Configuration
DATASET_CONFIGS = [ 
    {
        "path": "./datasets/GUI_Odyssey/train_600.jsonl",
        "sample_count": 600,
        "name": "GUI_Odyssey", 
    },
     {
        "path": "./datasets/GUIAct_Web/train_600.jsonl",
        "sample_count": 600,
        "name": "GUIAct_Web",
    },
    {
        "path": "./datasets/Mind2Web/train_600.jsonl",
        "sample_count": 600,  
        "name": "Mind2Web",   
    }
]

# Domain Mapping for Skew Modes
DOMAIN_MAPPING = {
    "Mobile": ["AC", "AitW", "GUI_Odyssey"],
    "Web": ["GUIAct_Web", "Mind2Web", "OmniAct_Web"],
    "Desktop": ["AS", "Omni_mac", "Omni_win"]
}

# =================================================================
# 🎯 Prompt Templates
# =================================================================

STEP_PROMPT_TEMPLATE = """
Your current task instruction, action history, and associated screenshot are as follows:
Screenshot: <image>
Task: {instruction}
History: 
{history}
"""

ALL_DATASETS_TEMPLATE = """\
You are a foundational action model capable of automating tasks across various digital environments...
(Template content preserved for functionality)
...
Actions: Specify the actual actions you will take based on your reasoning.
"""

PROMPT_TEMPLATES = {
    "default": ALL_DATASETS_TEMPLATE
}

# =================================================================
# ⚙️ Action Formatting
# =================================================================

def format_action(origin_action):
    """Convert raw action to standardized format"""
    if isinstance(origin_action, str):
        try:
            action = json.loads(origin_action)
        except json.JSONDecodeError:
            return origin_action
    else:
        action = origin_action
    
    if not isinstance(action, dict):
        return str(action)
    
    action_type = action.get("action_type", "").lower()
    
    if action_type == "click":
        x, y = action.get("x", 0), action.get("y", 0)
        return f"CLICK <point>[[{x}, {y}]]</point>"
    elif action_type == "type":
        text = action.get("input_text", "")
        return f"TYPE [{text}]"
    elif action_type == "scroll":
        direction = action.get("direction", "UP").upper()
        return f"SCROLL [{direction}]"
    elif action_type == "complete" or (action_type == "status" and action.get("goal_status") == "successful"):
        return "COMPLETE"
    elif action_type == "impossible" or (action_type == "status" and action.get("goal_status") == "impossible"):
        return "IMPOSSIBLE"
    elif action_type == "wait":
        return "WAIT"
    elif action_type == "long_press":
        x, y = action.get("x", 0), action.get("y", 0)
        return f"LONG_PRESS <point>[[{x}, {y}]]</point>"
    elif action_type == "navigate_back":
        return "NAVIGATE_BACK"
    elif action_type == "navigate_home":
        return "NAVIGATE_HOME"
    elif action_type == "open_app":
        app_name = action.get("app_name", "")
        return f"OPEN_APP [{app_name}]"
    elif action_type == "press_recent":
        return "PRESS_RECENT"
    elif action_type == "double_click":
        x, y = action.get("x", 0), action.get("y", 0)
        return f"DOUBLE_CLICK <point>[[{x}, {y}]]</point>"
    elif action_type == "right_click":
        x, y = action.get("x", 0), action.get("y", 0)
        return f"RIGHT_CLICK <point>[[{x}, {y}]]</point>"
    elif action_type == "moveto":
        x, y = action.get("x", 0), action.get("y", 0)
        return f"MOVETO <point>[[{x}, {y}]]</point>"
    elif action_type == "hotkey":
        keys = action.get("keys", "")
        return f"HOTKEY [{keys}]"
    elif action_type == "copy":
        text = action.get("text", "")
        return f"COPY [{text}]"
    elif action_type == "press_enter":
        return f"HOTKEY [ENTER]"
    else:
        return f"UNKNOWN_ACTION: {action}"

def format_history(history_steps):
    """Format history list"""
    return "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(history_steps)]) if history_steps else "None"

# =================================================================
# 📊 Global Settings
# =================================================================

SHOW_DETAILED_STATS = True
CONTINUE_ON_ERROR = True
PROGRESS_INTERVAL = 100

# =================================================================
# 🚀 Core Converter Class
# =================================================================

class EpisodeToStepConverter:
    def __init__(self, dataset_configs: List[Dict], num_clients: int):
        self.dataset_configs = dataset_configs
        self.num_clients = num_clients
        self.stats = {
            "total_episodes_read": 0,
            "total_steps_generated": 0,
            "total_errors": 0,
            "dataset_stats": {},
            "client_distribution": {}
        }
    
    def load_and_sample_episodes(self, config: Dict) -> List[Dict[str, Any]]:
        """Load and sample episodes from a single dataset file"""
        file_path, sample_count, dataset_name = config["path"], config["sample_count"], config["name"]
        
        print(f"Processing dataset: {dataset_name} | Path: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            self.stats["total_errors"] += 1
            if not CONTINUE_ON_ERROR:
                raise FileNotFoundError(f"File not found: {file_path}")
            return []
        
        try:
            all_episodes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            all_episodes.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {file_path} line {line_num}: {e}")
                            self.stats["total_errors"] += 1
            
            total_available = len(all_episodes)
            self.stats["total_episodes_read"] += total_available
            
            if total_available == 0:
                print(f"Warning: No valid data in {file_path}")
                return []
            
            if sample_count > total_available:
                print(f"Warning: sample_count ({sample_count}) > available ({total_available})")
                sample_count = total_available
            
            sampled_episodes = random.sample(all_episodes, sample_count)
            for episode in sampled_episodes:
                episode["dataset_source"] = dataset_name
            
            self.stats["dataset_stats"][dataset_name] = {
                "total_available": total_available,
                "sampled": len(sampled_episodes),
                "sample_rate": len(sampled_episodes) / total_available * 100
            }
            return sampled_episodes
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            self.stats["total_errors"] += 1
            if not CONTINUE_ON_ERROR: raise
            return []
    
    def convert_episode_to_steps(self, episode: Dict[str, Any], client_id: int) -> List[Dict[str, Any]]:
        """Convert single episode to multiple steps with prompts"""
        steps = []
        try:
            instruction = episode.get("instruction", "")
            acts_origin = episode.get("acts_origin", [])
            imgs = episode.get("imgs", [])
            dataset_source = episode.get("dataset_source", "")
            
            min_length = min(len(acts_origin), len(imgs))
            if min_length == 0: return []
            
            history_steps = []
            base_template = PROMPT_TEMPLATES.get(dataset_source, PROMPT_TEMPLATES["default"])
            
            for i in range(min_length):
                action, image_path = acts_origin[i], imgs[i]
                step_prompt = STEP_PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    history=format_history(history_steps)
                )
                formatted_action = format_action(action)
                
                steps.append({
                    "images": image_path,
                    "query": base_template + step_prompt,
                    "response": f"Actions:\n{formatted_action}",
                    "client_id": client_id,
                })
                history_steps.append(formatted_action)
        except Exception as e:
            print(f"Error converting episode: {e}")
            self.stats["total_errors"] += 1
        return steps
    
    def process_all_datasets(self) -> List[Dict[str, Any]]:
        """Collect and distribute episodes according to mode"""
        all_episodes = []
        for config in self.dataset_configs:
            all_episodes.extend(self.load_and_sample_episodes(config))
        
        if not all_episodes: return []
        
        if DISTRIBUTION_MODE == "iid-level":
            return self._process_iid_level(all_episodes)
        elif DISTRIBUTION_MODE == "skew-1":
            return self._process_skew_1(all_episodes)
        elif DISTRIBUTION_MODE == "skew-1-random":
            return self._process_skew_1_random_no_reuse(all_episodes, self._get_skew_1_mapping())
        elif DISTRIBUTION_MODE == "skew-device":
            return self._process_skew_device(all_episodes)
        elif DISTRIBUTION_MODE == "skew-device-random":
            return self._process_skew_device_random(all_episodes)
        elif DISTRIBUTION_MODE == "skew-dataset":
            return self._process_skew_dataset(all_episodes)
        else:
            return self._process_standard_modes(all_episodes)

    def _process_standard_modes(self, all_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_steps = []
        total_episodes = len(all_episodes)
        client_episodes = {cid: [] for cid in range(self.num_clients)}

        if DISTRIBUTION_MODE == "iid":
            for i, episode in enumerate(all_episodes):
                client_id = i % self.num_clients
                client_episodes[client_id].append(episode)
        elif DISTRIBUTION_MODE == "full-random":
            for episode in all_episodes:
                client_id = random.randint(0, self.num_clients - 1)
                client_episodes[client_id].append(episode)
        elif DISTRIBUTION_MODE == "iid-random":
            random.shuffle(all_episodes)
            split_size = total_episodes // self.num_clients
            for cid in range(self.num_clients):
                start = cid * split_size
                end = (cid + 1) * split_size if cid < self.num_clients - 1 else total_episodes
                client_episodes[cid] = all_episodes[start:end]

        for client_id in range(self.num_clients):
            for episode in client_episodes[client_id]:
                ds = episode.get("dataset_source", "unknown")
                self.stats["client_distribution"].setdefault(client_id, {}).setdefault(ds, {"episodes": 0, "steps": 0})
                steps = self.convert_episode_to_steps(episode, client_id)
                all_steps.extend(steps)
                self.stats["total_steps_generated"] += len(steps)
                self.stats["client_distribution"][client_id][ds]["episodes"] += 1
                self.stats["client_distribution"][client_id][ds]["steps"] += len(steps)
        return all_steps

    def _process_iid_level(self, all_episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        episodes_by_dataset = {}
        for ep in all_episodes:
            ds = ep.get("dataset_source", "unknown")
            episodes_by_dataset.setdefault(ds, []).append(ep)
        
        level_ranges = []
        cumulative = 0
        for count in IID_LEVEL_DISTRIBUTION:
            level_ranges.append((cumulative, cumulative + count))
            cumulative += count
        
        dataset_list = list(episodes_by_dataset.keys())
        client_episodes = {cid: [] for cid in range(self.num_clients)}
        
        for ds_idx, ds_name in enumerate(dataset_list):
            episodes = episodes_by_dataset[ds_name]
            for cid in range(self.num_clients):
                level_idx = (cid + ds_idx) % len(IID_LEVEL_DISTRIBUTION)
                start, end = level_ranges[level_idx]
                for ep in episodes[start:end]:
                    client_episodes[cid].append((ep, ds_name))
        
        return self._convert_client_episodes_to_steps(client_episodes)

    def _get_skew_1_mapping(self):
        group_size = self.num_clients // 3
        mapping = {}
        for cid in range(self.num_clients):
            if cid < group_size: mapping[cid] = ["Web", "Desktop"]
            elif cid < 2 * group_size: mapping[cid] = ["Mobile", "Desktop"]
            else: mapping[cid] = ["Mobile", "Web"]
        return mapping

    def _process_skew_1(self, all_episodes):
        return self._process_skew_1_no_reuse(all_episodes, self._get_skew_1_mapping())

    def _process_skew_device(self, all_episodes):
        group_size = self.num_clients // 3
        mapping = {}
        for cid in range(self.num_clients):
            if cid < group_size: mapping[cid] = ["Mobile"]
            elif cid < 2 * group_size: mapping[cid] = ["Web"]
            else: mapping[cid] = ["Desktop"]
        return self._process_skew_1_no_reuse(all_episodes, mapping)

    def _process_skew_1_no_reuse(self, all_episodes, client_domain_mapping):
        episodes_by_dataset = {}
        for ep in all_episodes:
            ds = ep.get("dataset_source", "unknown")
            episodes_by_dataset.setdefault(ds, []).append(ep)
        
        dataset_to_clients = {}
        for cid, domains in client_domain_mapping.items():
            for dom in domains:
                if dom in DOMAIN_MAPPING:
                    for ds in DOMAIN_MAPPING[dom]:
                        dataset_to_clients.setdefault(ds, []).append(cid)
        
        client_episodes = {cid: [] for cid in range(self.num_clients)}
        for ds_name, eps in episodes_by_dataset.items():
            if ds_name not in dataset_to_clients: continue
            targets = sorted(list(set(dataset_to_clients[ds_name])))
            num_t = len(targets)
            per_c = len(eps) // num_t
            for i, cid in enumerate(targets):
                start = i * per_c
                end = (i + 1) * per_c if i != num_t - 1 else len(eps)
                for ep in eps[start:end]:
                    client_episodes[cid].append((ep, ds_name))
        return self._convert_client_episodes_to_steps(client_episodes)

    def _convert_client_episodes_to_steps(self, client_episodes):
        all_steps = []
        for cid, eps_data in client_episodes.items():
            for ep, ds_name in (eps_data if isinstance(eps_data[0], tuple) else [(e, e['dataset_source']) for e in eps_data]):
                self.stats["client_distribution"].setdefault(cid, {}).setdefault(ds_name, {"episodes": 0, "steps": 0})
                steps = self.convert_episode_to_steps(ep, cid)
                all_steps.extend(steps)
                self.stats["total_steps_generated"] += len(steps)
                self.stats["client_distribution"][cid][ds_name]["episodes"] += 1
                self.stats["client_distribution"][cid][ds_name]["steps"] += len(steps)
        return all_steps

    def _process_skew_dataset(self, all_episodes):
        episodes_by_dataset = {}
        for ep in all_episodes:
            ds = ep.get("dataset_source", "unknown")
            episodes_by_dataset.setdefault(ds, []).append(ep)
        
        ds_list = sorted(episodes_by_dataset.keys())
        c_per_ds = self.num_clients // len(ds_list)
        client_episodes = {cid: [] for cid in range(self.num_clients)}
        
        for ds_idx, ds_name in enumerate(ds_list):
            eps = episodes_by_dataset[ds_name]
            for i in range(c_per_ds):
                target_cid = ds_idx * c_per_ds + i
                start = (len(eps) // c_per_ds) * i
                end = len(eps) if i == c_per_ds - 1 else (len(eps) // c_per_ds) * (i + 1)
                for ep in eps[start:end]:
                    client_episodes[target_cid].append((ep, ds_name))
        return self._convert_client_episodes_to_steps(client_episodes)

    def save_data(self, data_list: List[Dict[str, Any]], output_file: str):
        print(f"Saving JSONL to: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def save_data_as_json(self, data_list: List[Dict[str, Any]], output_file: str):
        print(f"Saving JSON array to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    def save_client_statistics(self, output_file: str):
        stats_file = os.path.splitext(output_file)[0] + "_client_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats["client_distribution"], f, indent=2)
        print(f"Stats saved to: {stats_file}")

    def run(self, output_file: str):
        random.seed(42)
        all_steps = self.process_all_datasets()
        if not all_steps: return
        self.save_data(all_steps, output_file)
        self.save_data_as_json(all_steps, os.path.splitext(output_file)[0] + ".json")
        self.save_client_statistics(output_file)
        print("Conversion Complete.")

def main():
    print("Episode to Step Converter Initialized")
    converter = EpisodeToStepConverter(DATASET_CONFIGS, NUM_CLIENTS)
    converter.run(OUTPUT_FILE)

if __name__ == "__main__":
    main()