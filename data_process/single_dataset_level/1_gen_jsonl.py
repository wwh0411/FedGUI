import os
import re
import json
from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom
import numpy as np
from tqdm import tqdm

# Note: Ensure these local utilities are provided in your repository
# from utils.io import load_json, load_xml, dump_jsonl

def read_xml_files(episode_id, base_path):
    """
    Reads all XML files for a given episode and returns their pretty-printed content.
    """
    episode_path = os.path.join(base_path, episode_id)
    xml_contents = {}

    # Sort files numerically by filename
    xml_files = sorted(os.listdir(episode_path), key=lambda x: int(os.path.splitext(x)[0]))

    for filename in xml_files:
        if filename.endswith('.xml'):
            file_path = os.path.join(episode_path, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            xml_str = ET.tostring(root, encoding='unicode', method='xml')
            dom = xml.dom.minidom.parseString(xml_str)
            xml_contents[os.path.splitext(filename)[0]] = dom.toprettyxml()

    return xml_contents

def find_index_for_coordinates(x, y, bounds_info):
    """
    Finds the index of the UI element whose bounds contain the point (x, y).
    Selects the smallest element if nested.
    """
    min_area = np.inf
    min_area_index = -1
    for index, info in bounds_info.items():
        bounds = info['bounds']
        if bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]:
            if info['area'] < min_area:
                min_area = info['area']
                min_area_index = index
    return min_area_index

def describe_action(action, bounds_info):
    """
    Generates a natural language description for structured actions.
    """
    if action["action_type"] in ["click", "long_press"]:
        act = action['action_type'].replace('_', ' ').capitalize()
        index = action['index']
        if index != -1 and bounds_info:
            node_desc = eval(bounds_info[index]['desc'])
            # Mapping attributes to natural language description
            if node_desc.get('text'):
                desc = f"{act} on button with text \"{node_desc['text']}\""
            elif node_desc.get('tooltip_text'):
                desc = f"{act} on button with text \"{node_desc['tooltip_text']}\""
            elif node_desc.get('content-desc'):
                desc = f"{act} on button with function \"{node_desc['content-desc']}\""
            else:
                desc = f"{act} on button"
        else:
            desc = f"{act} on button"
    elif action['action_type'] in ['open_app', 'wait', 'navigate_back', 'input_text', 'navigate_home']:
        if action['action_type'] == 'open_app':
            desc = f"Open App: {action.get('app_name', 'Unknown')}"
        elif action['action_type'] == 'input_text':
            desc = f"Type text: {action.get('text', '')}"
        else:
            desc = action['action_type'].replace('_', ' ').capitalize()
    elif action['action_type'] == 'scroll':
        desc = f"Scroll {action.get('direction', 'down')}"
    elif action['action_type'] == 'status':
        desc = f"Check status: {action.get('goal_status', 'unknown')}"
    else:
        raise ValueError(f"Unknown action type: {action['action_type']}")
    return desc

def process_episodes(data_dir: Path, episode_id_list=None):
    """
    Main processing loop for data conversion.
    """
    all_messages_step = []
    all_messages_episode = []
    
    # Locate all task metadata
    episode_paths = [x.parent for x in data_dir.rglob('task_info.json')]
    if episode_id_list:
        episode_paths = [x for x in episode_paths if int(x.stem) in episode_id_list]
    episode_paths = sorted(episode_paths)

    pattern = re.compile(r"^\d+$")

    for episode_path in tqdm(episode_paths):
        episode_id = str(episode_path.relative_to(data_dir))
        # from utils.io import load_json
        task_info = load_json(episode_path / 'task_info.json')
        
        actions = task_info['actions']
        goal = task_info['goal']
        sub_goals = task_info['sub_goal'] + ['Check if the task is finished']
        
        screenshots = sorted([i for i in episode_path.glob("*.png") if pattern.match(i.stem)], key=lambda x: int(x.stem))
        xmls = [x.with_suffix('.xml') for x in screenshots]
        lightxmls = [x.with_suffix('.lightxml') for x in screenshots]

        actions_convert = []

        for step_idx, (xml, lightxml, screenshot, action, sub_goal) in enumerate(zip(xmls, lightxmls, screenshots, actions, sub_goals)):
            action_json = json.loads(action)
            action_convert = action # Fallback
            
            if action_json["action_type"] in ["click", "long_press"]:
                try:
                    # from utils.io import load_xml
                    # raw_tree is used for coordinate decoding, light_tree for metadata
                    raw_tree, _ = load_xml(xml, pretty=True)
                    light_tree, _ = load_xml(lightxml, pretty=True)
                    
                    # Logic to decode bounds (similar to previous scripts)
                    # index2bounds = decode_bounds_from_tree(raw_tree)
                    # bounds_info = extract_bounds_in_tree(light_tree, index2bounds)
                    
                    # For this desensitized version, assume bounds_info extraction logic exists
                    bounds_info = {} 
                    
                    x, y = action_json["x"], action_json["y"]
                    index = find_index_for_coordinates(x, y, bounds_info)
                    action_w_index = {"action_type": action_json["action_type"], "index": index}
                    action_convert = describe_action(action_w_index, bounds_info)
                except:
                    action_convert = action
            else:
                action_convert = describe_action(action_json, bounds_info=None)

            actions_convert.append(action_convert)

            all_messages_step.append({
                "episode_id": episode_id,
                "instruction": goal,
                "sub_instruction": sub_goal,
                "act_origin": action,
                "act_convert": action_convert,
                "img": str(screenshot),
                "lightxml": str(lightxml)
            })

        all_messages_episode.append({
            "episode_id": episode_id,
            "instruction": goal,
            "acts_convert": actions_convert,
            "imgs": [str(x) for x in screenshots]
        })

    # Save outputs
    # from utils.io import dump_jsonl
    dump_jsonl(all_messages_step, data_dir.parent / "step-wise-all.jsonl")
    dump_jsonl(all_messages_episode, data_dir.parent / "episode-wise-all.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Normalize UI interactions to text descriptions.")
    parser.add_argument('--data_dir', type=str, default='./data/processed_android_control', help='Unpacked data root.')
    args = parser.parse_args()

    process_episodes(Path(args.data_dir))

if __name__ == '__main__':
    main()