import json
import re
import os
import sys
import argparse

def convert_pyautogui_to_json(action_str, stats_dict=None):
    """
    Converts PyAutoGUI formatted operations into a standardized JSON format.
    
    Args:
        action_str (str): PyAutoGUI formatted action string.
        stats_dict (dict): Dictionary for tracking action counts.
        
    Returns:
        str: JSON formatted action string.
    """
    action_str = action_str.strip()
    
    # Return immediately if already in JSON format
    if action_str.startswith('{"action_type"'):
        return action_str
    
    action_str_lower = action_str.lower()
    
    # 1. Click: pyautogui.click(x,y) -> {"action_type":"click","x":x,"y":y}
    click_match = re.match(r'(?:py)?autogui\.click\(([^,]+),\s*([^)]+)\)', action_str_lower)
    if click_match:
        if stats_dict is not None:
            stats_dict['click'] = stats_dict.get('click', 0) + 1
        x, y = click_match.groups()
        x = re.sub(r'[^0-9.-]', '', x.strip())
        y = re.sub(r'[^0-9.-]', '', y.strip())
        try:
            return json.dumps({"action_type": "click", "x": float(x), "y": float(y)}, ensure_ascii=False)
        except ValueError:
            if stats_dict is not None:
                stats_dict['coordinate_error'] = stats_dict.get('coordinate_error', 0) + 1
            print(f"Coordinate parse error in click: x='{x}', y='{y}', raw: {action_str}")
            return json.dumps({"action_type": "unknown", "original": action_str, "error": "coordinate_parse_error"}, ensure_ascii=False)
    
    # 2. rightClick: pyautogui.rightClick(x,y) -> {"action_type":"rightclick","x":x,"y":y}
    right_click_match = re.match(r'(?:py)?autogui\.rightclick\(([^,]+),\s*([^)]+)\)', action_str_lower)
    if right_click_match:
        if stats_dict is not None:
            stats_dict['rightclick_with_params'] = stats_dict.get('rightclick_with_params', 0) + 1
        x, y = right_click_match.groups()
        x = re.sub(r'[^0-9.-]', '', x.strip())
        y = re.sub(r'[^0-9.-]', '', y.strip())
        try:
            return json.dumps({"action_type": "rightclick", "x": float(x), "y": float(y)}, ensure_ascii=False)
        except ValueError:
            print(f"Coordinate parse error in rightclick: x='{x}', y='{y}'")
            return json.dumps({"action_type": "unknown", "original": action_str}, ensure_ascii=False)
    
    # 3. rightClick without parameters -> Delete this episode
    if re.match(r'(?:py)?autogui\.rightclick\(\)', action_str_lower):
        if stats_dict is not None:
            stats_dict['rightclick_no_params_deleted'] = stats_dict.get('rightclick_no_params_deleted', 0) + 1
        print(f"Deleted rightClick with no params: {action_str}")
        return json.dumps({"action_type": "DELETE_EPISODE", "original": action_str}, ensure_ascii=False)
    
    # 4. scroll: pyautogui.scroll(amount)
    scroll_match = re.match(r'(?:py)?autogui\.scroll\(([^)]+)\)', action_str_lower)
    if scroll_match:
        if stats_dict is not None:
            stats_dict['scroll'] = stats_dict.get('scroll', 0) + 1
        amount = int(scroll_match.group(1))
        return json.dumps({"action_type": "scroll", "direction": "up" if amount > 0 else "down"}, ensure_ascii=False)
    
    # 5. write: pyautogui.write("text")
    write_match = re.match(r'(?:py)?autogui\.write\("([^"]*)"\)', action_str_lower)
    if write_match:
        if stats_dict is not None:
            stats_dict['write'] = stats_dict.get('write', 0) + 1
        return json.dumps({"action_type": "type", "input_text": write_match.group(1)}, ensure_ascii=False)
    
    # 6. hotkey patterns
    hotkey_match = re.match(r'(?:py)?autogui\.hotkey\("([^"]*)",\s*"([^"]*)"\)', action_str_lower)
    if hotkey_match:
        if stats_dict is not None:
            stats_dict['hotkey_2keys'] = stats_dict.get('hotkey_2keys', 0) + 1
        return json.dumps({"action_type": "hotkey", "button": f"{hotkey_match.group(1)}+{hotkey_match.group(2)}"}, ensure_ascii=False)
    
    # 7. moveTo
    move_to_match = re.match(r'(?:py)?autogui\.moveto\(([^,]+),\s*([^)]+)\)', action_str_lower)
    if move_to_match:
        x, y = [re.sub(r'[^0-9.-]', '', val.strip()) for val in move_to_match.groups()]
        try:
            return json.dumps({"action_type": "moveto", "x": float(x), "y": float(y)}, ensure_ascii=False)
        except ValueError:
            return json.dumps({"action_type": "unknown", "original": action_str}, ensure_ascii=False)

    print(f"Warning: Unrecognized action format: {action_str}")
    return json.dumps({"action_type": "unknown", "original": action_str}, ensure_ascii=False)

def parse_task_file(task_file_path, base_path):
    """
    Parses task description files to extract instructions and actions.
    """
    if task_file_path.startswith('data/tasks/'):
        actual_path = task_file_path.replace('data/tasks/', os.path.join(base_path, 'data/tasks/'))
    else:
        actual_path = task_file_path
    
    try:
        with open(actual_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Failed to read task file: {actual_path}, error: {e}")
        return None, None
    
    instruction = ''
    actions = []
    parse_action = False
    has_output_script = any('output script:' in line.lower() for line in lines)
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('task:'):
            instruction = line[len('Task:'):].strip()
            continue
        if has_output_script:
            if line.lower().startswith('output script:'):
                parse_action = True
                content = line[len('Output Script:'):].strip()
                if content: actions.append(content)
                continue
            if parse_action and line:
                actions.append(line)
        else:
            if line: actions.append(line)
    
    return instruction, actions

def process_omniact_complete(input_json_file, output_jsonl_file, base_path):
    """
    Complete processing from raw OmniAct JSON to target JSONL format.
    """
    print("Starting OmniAct complete processing...")
    
    stats = {"total": 0, "processed": 0, "converted": 0, "deleted": 0, "errors": 0}
    action_stats = {}
    
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats["total"] = len(data)
    print(f"Total episodes found: {stats['total']}")
    
    with open(output_jsonl_file, 'w', encoding='utf-8') as fout:
        for idx, item in data.items():
            try:
                instruction, actions = parse_task_file(item['task'], base_path)
                if instruction is None or not actions:
                    stats["errors"] += 1
                    continue
                
                first_action = actions[0]
                stats["processed"] += 1
                
                # Action conversion logic
                try:
                    try:
                        action_obj = json.loads(first_action)
                    except json.JSONDecodeError:
                        converted_str = convert_pyautogui_to_json(first_action, action_stats)
                        action_obj = json.loads(converted_str)
                        stats["converted"] += 1
                        
                        if action_obj.get('action_type') == 'DELETE_EPISODE':
                            stats["deleted"] += 1
                            continue
                except Exception as e:
                    stats["errors"] += 1
                    continue
                
                # Normalize image path to absolute path
                image_path = item['image']
                absolute_image_path = os.path.join(base_path, image_path) if image_path.startswith('data/') else image_path
                
                # Construct final item
                output_item = {
                    'episode_id': f'omniact_{idx}',
                    'instruction': instruction,
                    'acts_origin': [action_obj],
                    'imgs': [absolute_image_path],
                }
                
                fout.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error at Episode {idx}: {e}")
                stats["errors"] += 1
                continue
    
    print(f"\nProcessing Complete!")
    print(f"Processed: {stats['processed']} | Converted: {stats['converted']} | Deleted: {stats['deleted']} | Errors: {stats['errors']}")

def main():
    parser = argparse.ArgumentParser(description='Process OmniAct dataset and convert PyAutoGUI to JSON format.')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path (e.g., test.json)')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file path')
    parser.add_argument('--base_path', '-b', required=True, help='Base directory of OmniAct dataset')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not os.path.exists(args.base_path):
        print(f"Error: Base path not found: {args.base_path}")
        return
    
    process_omniact_complete(args.input, args.output, args.base_path)
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default placeholder config for local development
        print("Using default placeholder configuration...")
        DEFAULT_INPUT = "./data/omniact/raw/test.json"
        DEFAULT_OUTPUT = "./data/omniact/processed/test.jsonl"
        DEFAULT_BASE = "./data/omniact/"
        
        if os.path.exists(DEFAULT_INPUT):
            process_omniact_complete(DEFAULT_INPUT, DEFAULT_OUTPUT, DEFAULT_BASE)
        else:
            print("Default input not found. Please use command line arguments.")
    else:
        main()