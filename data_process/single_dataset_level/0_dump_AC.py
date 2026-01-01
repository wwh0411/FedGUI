import os.path
import tensorflow as tf
import cv2
import pickle as pkl
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from utils.decode import decode_image, decode_tree, convert_win_to_xml
from utils.node_process import extract_nodes, dump_nodes_to_xml, is_point_in_node
from utils.bbox import calculate_iof
from tqdm import tqdm

def dump_one_episode_observation(screenshots, screenshot_widths, screenshot_heights, forests,
                                 actions, out_ep_dir: Path):
    """
    Saves screenshot images and parses accessibility trees into XML files for a single episode.
    """
    for step_id, (screenshot, w, h, forest, action) in enumerate(zip(
            screenshots, screenshot_widths, screenshot_heights, forests, actions)):
        
        out_file = out_ep_dir / f'{step_id:02d}.xml'
        
        # Convert RGBA to BGR for OpenCV saving
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(out_file.with_suffix('.png')), screenshot_bgr)
        
        # Parse action and extract UI nodes
        action_dict = eval(action)
        all_node_list = extract_nodes(forest.windows, h, w)
        
        # Optional: Validation logic to ensure action coordinates match a UI node
        # if action_dict['action_type'] in ['click', 'long_press']:
        #     check_list = [is_point_in_node((action_dict['x'], action_dict['y']), node) for node in all_node_list]
        #     assert sum(check_list) > 0, 'No matching node found for the action.'

        try:
            dump_nodes_to_xml(all_node_list, out_file, out_file.with_suffix('.lightxml'))
        except Exception:
            print(f'Warning: Incorrect XML parsing for episode at {out_ep_dir}')

def dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir):
    """
    Saves episode-level metadata and task instructions to a JSON file.
    """
    annotations = {
        'goal': goal,
        'actions': actions,
        'sub_goal': step_instructions,
    }
    with open(out_ep_dir / 'task_info.json', 'w') as json_file:
        json.dump(annotations, json_file)

def dump_all_episodes(dataset, out_root_dir: Path):
    """
    Iterates through the TFRecord dataset to unpack and save all episodes.
    """
    episodes_step_instructions = defaultdict(list)
    
    for d in tqdm(dataset):
        ep = tf.train.Example()
        ep.ParseFromString(d)

        # Extract basic episode information
        ep_id = ep.features.feature['episode_id'].int64_list.value[0]
        step_instructions = [x.decode('utf-8') for x in
                             ep.features.feature['step_instructions'].bytes_list.value]
        
        # Setup output directory for the episode
        out_ep_dir = out_root_dir / f'{ep_id:06d}'
        out_ep_dir.mkdir(exist_ok=True, parents=True)
        
        # Skip if already processed
        if (out_ep_dir / f'{len(step_instructions):02d}.png').exists():
            continue
            
        # Decode sequential features
        goal = ep.features.feature['goal'].bytes_list.value[0].decode('utf-8')
        screenshots = [decode_image(x) for x in ep.features.feature['screenshots'].bytes_list.value]
        screenshot_widths = [x for x in ep.features.feature['screenshot_widths'].int64_list.value]
        screenshot_heights = [x for x in ep.features.feature['screenshot_heights'].int64_list.value]
        actions = [x.decode('utf-8') for x in ep.features.feature['actions'].bytes_list.value]
        forests = [decode_tree(x) for x in ep.features.feature['accessibility_trees'].bytes_list.value]

        assert ep_id not in episodes_step_instructions, f'Episode {ep_id} has already been processed'
        episodes_step_instructions[ep_id].append(step_instructions)
        
        # Append completion status
        actions.append("{\"action_type\":\"status\",\"goal_status\":\"successful\"}")
        
        try:
            dump_one_episode_observation(screenshots, screenshot_widths, screenshot_heights, forests,
                                          actions, out_ep_dir)
            dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir)

        except Exception as e:
            import traceback
            err_str = traceback.format_exc()
            print(f"Error in episode {ep_id}:\n{err_str}")
            
            error_log_file = out_root_dir / 'error_log.txt'
            with open(error_log_file, 'a') as file:
                file.write(f'[{ep_id}]: {str(e)}\n')

        if len(episodes_step_instructions) % 200 == 0:
            print(f'Progress: Read {len(episodes_step_instructions)} episodes.')
            
    return episodes_step_instructions

if __name__ == "__main__":
    # Define placeholder paths for repository use
    INPUT_TFRECORD_DIR = "./data/raw_tfrecords/*"
    OUTPUT_UNPACK_DIR = Path("./data/unpacked_episodes")

    # Load and sort TFRecord files
    tfrecord_files = tf.io.gfile.glob(INPUT_TFRECORD_DIR)
    tfrecord_files = sorted(tfrecord_files)[:20] # Limit for initial testing
    
    if not tfrecord_files:
        print(f"No files found at {INPUT_TFRECORD_DIR}. Please check the data directory.")
    else:
        # Initialize TFRecord iterator (GZIP compressed)
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type='GZIP').as_numpy_iterator()
        
        print(f"Starting to unpack episodes to {OUTPUT_UNPACK_DIR}...")
        dump_all_episodes(raw_dataset, OUTPUT_UNPACK_DIR)
        print("Unpacking complete.")