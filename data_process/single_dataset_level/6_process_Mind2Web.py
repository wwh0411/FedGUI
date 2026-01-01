#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mind2Web Data Conversion Tool
Function: Converts Mind2Web training data from JSON to JSONL format.
- annotation_id -> episode_id
- confirmed_task -> instruction  
- screenshot paths -> imgs (list format)
- Processes actions: click (coordinates), type (text), select, etc.
"""

import json
from collections import defaultdict
import os

def print_action_conversion_table():
    """
    Prints the action conversion table showing all supported action types and rules.
    """
    print("=" * 100)
    print("Mind2Web Action Conversion Table")
    print("=" * 100)
    
    headers = ["Original Op", "Converted Type", "Output Format", "Required Fields", "Coord Source"]
    rows = [
        ["CLICK", "click", '{"action_type": "click", "x": x, "y": y}', "x, y", "Bounding Box Center"],
        ["TYPE", "type", '{"action_type": "type", "input_text": "text"}', "input_text", "operation.value"],
        ["SELECT", "select_text", '{"action_type": "select_text", ...}', "from/to coords", "Bounding Box Corners"],
        ["ENTER", "hotkey", '{"action_type": "hotkey", "keys": "ENTER"}', "keys", "N/A"],
        ["HOVER", "wait", '{"action_type": "wait", "x": x, "y": y}', "x, y", "Bounding Box Center"]
    ]
    
    # Simple table printing logic
    print(f"{' | '.join(headers)}")
    print("-" * 100)
    for row in rows:
        print(f"{' | '.join(row)}")
    print("=" * 100)

def parse_bounding_box_for_select(bbox_str):
    """
    Parses 'x,y,width,height' string for SELECT actions.
    Returns: (from_coord, to_coord) as (top-left, bottom-right).
    """
    try:
        coords = bbox_str.split(',')
        if len(coords) == 4:
            x, y, w, h = map(float, coords)
            return (int(x), int(y)), (int(x + w), int(y + h))
    except:
        pass
    return None, None

def parse_bounding_box(bbox_str):
    """
    Parses 'x,y,width,height' string to return the center point.
    """
    try:
        coords = bbox_str.split(',')
        if len(coords) == 4:
            x, y, w, h = map(float, coords)
            return int(x + w / 2), int(y + h / 2)
    except:
        pass
    return None, None

def extract_action(item, stats):
    """
    Extracts structured action information from a data item.
    """
    try:
        operation = json.loads(item['operation'])
        original_op = operation['original_op']
        value = operation.get('value', '')
        
        # Determine coordinate info from pos_candidates
        bbox = None
        if item.get('pos_candidates'):
            candidate = json.loads(item['pos_candidates'][0])
            attributes = json.loads(candidate.get('attributes', '{}'))
            bbox = attributes.get('bounding_box_rect')

        if original_op == 'CLICK':
            x, y = parse_bounding_box(bbox) if bbox else (None, None)
            return {"action_type": "click", "x": x or 0, "y": y or 0}
        
        elif original_op == 'TYPE':
            return {"action_type": "type", "input_text": value}
            
        elif original_op == 'SELECT':
            from_c, to_c = parse_bounding_box_for_select(bbox) if bbox else (None, None)
            return {"action_type": "select_text", "from_coordinate": from_c or (0,0), "to_coordinate": to_c or (0,0)}
            
        elif original_op == 'ENTER':
            return {"action_type": "hotkey", "keys": "ENTER"}
            
        elif original_op == 'HOVER':
            x, y = parse_bounding_box(bbox) if bbox else (None, None)
            return {"action_type": "wait", "x": x or 0, "y": y or 0}
            
        return {"action_type": "unknown", "original_op": original_op}
            
    except Exception as e:
        return {"action_type": "error", "message": str(e)}

def convert_to_jsonl_with_actions(input_file, output_file):
    """
    Main conversion logic: JSON -> JSONL with processed actions and path cleaning.
    """
    print(f"Reading input file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by annotation_id (episode)
    grouped = defaultdict(list)
    for item in data:
        grouped[item['annotation_id']].append(item)
    
    converted_data = []
    
    for annotation_id, items in grouped.items():
        # Sort items by action_uid to maintain sequence
        sorted_items = sorted(items, key=lambda x: x['action_uid'])
        
        imgs, actions = [], []
        valid_episode = True
        
        for item in sorted_items:
            # Check for valid screenshot path
            screenshot = item.get('screenshot')
            if not (screenshot and screenshot.get('path')):
                valid_episode = False
                break
            
            imgs.append(screenshot['path'])
            actions.append(extract_action(item, {}))
        
        if valid_episode and len(imgs) == len(actions):
            converted_data.append({
                "episode_id": annotation_id,
                "instruction": sorted_items[0]['confirmed_task'],
                "acts_origin": actions,
                "imgs": imgs
            })
    
    # Write to JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Conversion complete. Saved {len(converted_data)} episodes to {output_file}")
    return converted_data

def main():
    # Placeholder relative paths for the repository
    INPUT_FILE = "./data/mind2web/raw/test_website.json"
    OUTPUT_FILE = "./data/mind2web/processed/test_website.jsonl"
    
    if os.path.exists(INPUT_FILE):
        convert_to_jsonl_with_actions(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Error: Input file not found: {INPUT_FILE}")

if __name__ == "__main__":
    main()