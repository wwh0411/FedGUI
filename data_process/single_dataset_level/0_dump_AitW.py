import os.path
import tensorflow as tf
import cv2
import pickle as pkl
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Note: Ensure these utility functions are available in your local 'utils' directory
# from utils.decode import decode_image, decode_tree
# from utils.node_process import extract_nodes, dump_nodes_to_xml, is_point_in_node

def decode_image(byte_string):
    """Decodes image bytes using TensorFlow."""
    image = tf.io.decode_image(byte_string, channels=None, dtype=tf.dtypes.uint8, expand_animations=True)
    return image.numpy()

def _decode_image_raw(example, image_height, image_width, image_channels):
    """
    Decodes raw image bytes from a TFRecord example.
    
    Returns:
        Reshaped image numpy array (H, W, C).
    """
    image = tf.io.decode_raw(
        example.features.feature['image/encoded'].bytes_list.value[0],
        out_type=tf.uint8,
    )
    height = tf.cast(image_height, tf.int32)
    width = tf.cast(image_width, tf.int32)
    n_channels = tf.cast(image_channels, tf.int32)
    return tf.reshape(image, (height, width, n_channels))

def dump_one_episode_observations(screenshots, screenshot_widths, screenshot_heights, forests,
                                   actions, out_ep_dir: Path):
    """Saves screenshots and UI hierarchy (XML) for each step in an episode."""
    for step_id, (img, w, h, forest, action) in enumerate(zip(
            screenshots, screenshot_widths, screenshot_heights, forests, actions)):
        
        # Convert RGBA to BGR for OpenCV saving
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        action_dict = eval(action)
        
        # UI Node extraction logic (requires project utils)
        all_node_list = extract_nodes(forest.windows, h, w)
        
        # Validate if the action coordinate lands on a UI node
        if action_dict['action_type'] in ['click', 'long_press']:
            check_list = [is_point_in_node((action_dict['x'], action_dict['y']), node) for node in all_node_list]
            assert sum(check_list) > 0, f'Error: No matching UI node found for action at step {step_id}'
        
        out_file = out_ep_dir / f'{step_id:02d}.xml'
        dump_nodes_to_xml(all_node_list, out_file, out_file.with_suffix('.lightxml'))
        cv2.imwrite(str(out_file.with_suffix('.png')), img_bgr)

def dump_one_episode_annotations(goal, actions, step_instructions, out_ep_dir):
    """Saves task-level annotations to a JSON file."""
    annotations = {
        'goal': goal,
        'actions': actions,
        'sub_goal': step_instructions,
    }
    with open(out_ep_dir / 'task_info.json', 'w') as json_file:
        json.dump(annotations, json_file)

def print_tfrecord_format(files):
    """Helper to inspect the schema of the TFRecord files."""
    ds = tf.data.TFRecordDataset(files).batch(1)
    for batch_data in ds.take(1):
        for serialized_example in batch_data:
            example_proto = tf.train.Example.FromString(serialized_example.numpy())
            for key, feature in example_proto.features.feature.items():
                ftype = None
                if feature.HasField('bytes_list'): ftype = 'bytes_list'
                elif feature.HasField('float_list'): ftype = 'float_list'
                elif feature.HasField('int64_list'): ftype = 'int64_list'
                
                if ftype:
                    print(f'{key} : {ftype}')

def dump_all_episodes(dataset, out_root_dir: Path):
    """Main loop to iterate over TFRecord dataset and unpack episodes."""
    ep_step_record = defaultdict(list)
    
    for d in tqdm(dataset):
        example = tf.train.Example()
        example.ParseFromString(d)

        # Extract basic metadata
        ep_id = example.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
        ep_len = example.features.feature['episode_length'].int64_list.value[0]
        
        # Image processing
        h = example.features.feature['image/height'].int64_list.value[0]
        w = example.features.feature['image/width'].int64_list.value[0]
        c = example.features.feature['image/channels'].int64_list.value[0]
        
        image_tensor = _decode_image_raw(example, h, w, c)
        img_np = cv2.cvtColor(image_tensor.numpy(), cv2.COLOR_RGBA2BGR)
        
        # Setup output directory
        out_ep_dir = out_root_dir / f'{ep_id}'
        out_ep_dir.mkdir(exist_ok=True, parents=True)
        
        # Data parsing logic (Note: Modify according to specific AitW subset schema)
        try:
            # Example logic for unpacking sequence-based features
            # screenshots = [decode_image(x) for x in example.features.feature['screenshots'].bytes_list.value]
            # ... additional logic ...
            pass
        except Exception as e:
            import traceback
            print(f"Error processing episode {ep_id}: {e}")
            with open(out_root_dir / 'error_log.txt', 'a') as f_err:
                f_err.write(f'[{ep_id}]: {traceback.format_exc()}\n')

    return ep_step_record

if __name__ == "__main__":
    # Define generic paths for repository
    INPUT_PATTERN = "./data/aitw/general/*"
    OUTPUT_DIR = Path("./data/unpacked_aitw")

    # Fetch filenames using glob
    filenames = tf.io.gfile.glob(INPUT_PATTERN)
    
    if not filenames:
        print(f"No files found at {INPUT_PATTERN}. Please check data path.")
    else:
        # Load GZIP compressed TFRecords
        raw_iterator = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
        
        print(f"Starting to unpack episodes to {OUTPUT_DIR}...")
        ep_dict = dump_all_episodes(raw_iterator, OUTPUT_DIR)