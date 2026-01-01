#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Action Format Normalization Tool
Function: Standardizes action formats in raw dataset files for consistency.
"""

import json
import os
import shutil
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Dataset Configurations
DATASET_CONFIGS = [
    {
        "path": "./data/raw/android_sample.jsonl",
        "name": "General_Agent_Dataset",
        "backup_suffix": "_backup"
    },
    # Additional datasets can be added here using the same structure
]

# Mapping rules for action types
ACTION_TYPE_MAPPING = {
    "doubleclick": "double_click",
    "rightclick": "right_click",
    "press_button": "hotkey",
    "PRESS_BUTTON": "hotkey",
    "press_enter": "hotkey",
}

def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a single action dictionary based on predefined rules.
    """
    if not isinstance(action, dict):
        return action
    
    normalized_action = action.copy()
    action_type = action.get("action_type", "")
    
    # Apply type mapping
    if action_type in ACTION_TYPE_MAPPING:
        new_action_type = ACTION_TYPE_MAPPING[action_type]
        normalized_action["action_type"] = new_action_type
        
        # Handle press_button to hotkey conversion
        if action_type in ["press_button", "PRESS_BUTTON"]:
            if "button" in normalized_action:
                button = normalized_action.pop("button")
                normalized_action["keys"] = button.upper() if button else "ENTER"
        
        # Handle press_enter to hotkey conversion
        elif action_type == "press_enter":
            normalized_action["keys"] = "ENTER"
    
    # Ensure hotkey actions use 'keys' field instead of 'button'
    if normalized_action.get("action_type") == "hotkey" and "button" in normalized_action and "keys" not in normalized_action:
        button_value = normalized_action.pop("button")
        normalized_action["keys"] = button_value.upper()
    
    # Ensure keys are uppercase in hotkey actions
    if normalized_action.get("action_type") == "hotkey":
        if "keys" in normalized_action:
            keys_value = normalized_action["keys"]
            if isinstance(keys_value, str):
                normalized_action["keys"] = keys_value.upper()
        
        # Compatibility: rename 'key' to 'keys'
        if "key" in normalized_action:
            key_value = normalized_action.pop("key")
            if isinstance(key_value, str):
                normalized_action["keys"] = key_value.upper()

    # Standardize action_type to lowercase
    if "action_type" in normalized_action:
        normalized_action["action_type"] = normalized_action["action_type"].lower()
    
    return normalized_action

class DatasetNormalizer:
    def __init__(self, create_backup=True, dry_run=False):
        """
        Args:
            create_backup: Whether to create a backup file before modification.
            dry_run: If True, simulate changes without writing to disk.
        """
        self.create_backup = create_backup
        self.dry_run = dry_run
        self.dataset_configs = DATASET_CONFIGS
        self.stats = {
            "total_files": 0,
            "total_episodes": 0,
            "total_actions": 0,
            "modified_actions": 0,
            "dataset_stats": {},
            "errors": []
        }
    
    def backup_file(self, file_path: str, backup_suffix: str = "_backup") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}{backup_suffix}_{timestamp}.jsonl"
        
        if not self.dry_run:
            shutil.copy2(file_path, backup_path)
            print(f"   💾 Backup created: {backup_path}")
        else:
            print(f"   💾 [Simulate] Backup would be created: {backup_path}")
        
        return backup_path
    
    def normalize_episode(self, episode: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        modified_count = 0
        normalized_episode = episode.copy()
        
        if "acts_origin" in episode:
            normalized_acts = []
            for action in episode["acts_origin"]:
                original_action = action
                normalized_action = normalize_action(action)
                normalized_acts.append(normalized_action)
                
                if original_action != normalized_action:
                    modified_count += 1
            
            normalized_episode["acts_origin"] = normalized_acts
        
        return normalized_episode, modified_count
    
    def normalize_dataset_file(self, config: Dict[str, str]) -> Dict[str, Any]:
        file_path = config["path"]
        dataset_name = config["name"]
        backup_suffix = config.get("backup_suffix", "_backup")
        
        print(f"\n🔧 Processing Dataset: {dataset_name}")
        print(f"   Path: {file_path}")
        
        if not os.path.exists(file_path):
            error_msg = f"❌ File not found: {file_path}"
            print(error_msg)
            self.stats["errors"].append(error_msg)
            return {"episodes": 0, "actions": 0, "modified": 0}
        
        if self.create_backup:
            self.backup_file(file_path, backup_suffix)
        
        try:
            episodes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if line:
                            episodes.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        error_msg = f"⚠️ JSON decode error at {file_path} line {line_num}: {e}"
                        print(error_msg)
                        self.stats["errors"].append(error_msg)
            
            print(f"   📊 Loaded {len(episodes)} episodes")
            
            normalized_episodes = []
            total_actions = 0
            total_modified = 0
            
            for episode in episodes:
                normalized_episode, modified_count = self.normalize_episode(episode)
                normalized_episodes.append(normalized_episode)
                if "acts_origin" in episode:
                    total_actions += len(episode["acts_origin"])
                total_modified += modified_count
            
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for episode in normalized_episodes:
                        json.dump(episode, f, ensure_ascii=False)
                        f.write('\n')
                print(f"   ✅ Normalized data written to file")
            else:
                print(f"   ✅ [Simulate] Normalized data would be written")
            
            stats = {"episodes": len(episodes), "actions": total_actions, "modified": total_modified}
            print(f"   📈 Stats: {stats['episodes']} eps, {stats['actions']} acts, {stats['modified']} modified")
            return stats
            
        except Exception as e:
            error_msg = f"❌ Error processing {file_path}: {e}"
            print(error_msg)
            self.stats["errors"].append(error_msg)
            return {"episodes": 0, "actions": 0, "modified": 0}
    
    def normalize_all_datasets(self):
        print("🚀 Starting normalization of action formats...")
        if self.dry_run:
            print("🔍 [SIMulation Mode] - Files will not be modified")
        print("=" * 80)
        
        self.stats["total_files"] = len(self.dataset_configs)
        for config in self.dataset_configs:
            dataset_stats = self.normalize_dataset_file(config)
            self.stats["total_episodes"] += dataset_stats["episodes"]
            self.stats["total_actions"] += dataset_stats["actions"]
            self.stats["modified_actions"] += dataset_stats["modified"]
            self.stats["dataset_stats"][config["name"]] = dataset_stats
        
        self.print_summary()

    def print_summary(self):
        print("\n" + "=" * 80)
        print("📊 Normalization Summary")
        print("=" * 80)
        print(f"Files Processed: {self.stats['total_files']}")
        print(f"Total Episodes: {self.stats['total_episodes']}")
        print(f"Total Actions: {self.stats['total_actions']}")
        print(f"Modified Actions: {self.stats['modified_actions']}")
        if self.stats["total_actions"] > 0:
            rate = (self.stats["modified_actions"] / self.stats["total_actions"]) * 100
            print(f"Modification Rate: {rate:.2f}%")
        
        if self.stats["errors"]:
            print(f"\n❌ Errors found: {len(self.stats['errors'])}")
        
        if not self.dry_run:
            print(f"\n✅ Processing complete!")
        else:
            print("\n⚠️ Simulation complete. No files were changed.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Action Format Normalization Tool")
    parser.add_argument("--no-backup", action="store_true", help="Disable file backup")
    parser.add_argument("--execute", action="store_true", help="Execute actual modifications")
    parser.add_argument("--dataset", type=str, help="Process a specific dataset name")
    
    args = parser.parse_args()
    
    create_backup = not args.no_backup
    dry_run = not args.execute
    
    configs_to_process = DATASET_CONFIGS
    if args.dataset:
        configs_to_process = [c for c in DATASET_CONFIGS if c["name"].lower() == args.dataset.lower()]
    
    normalizer = DatasetNormalizer(create_backup=create_backup, dry_run=dry_run)
    normalizer.dataset_configs = configs_to_process
    normalizer.normalize_all_datasets()

if __name__ == "__main__":
    main()