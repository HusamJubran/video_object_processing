import configargparse
import os
import multiprocessing as mp
from .preprocess_1st_stage_per_video import Animal, pil_resize, Trajectory, save_trajectory_2nd_stage, Video_ID

import cv2
import json
from PIL import Image
import sys
from tqdm import tqdm
import random

# sys.path.append("/viscam/projects/video_animals/husam/prev_repos/MagicPony") 

file_types = ['full_mask.png', 'metadata.json', 'occlusion.png']




def get_video_path(input_path):
    """
    convert /..../all_Bb8nquNkruY_clips_after_1st_stage to /..../Bb8nquNkruY/Bb8nquNkruY.mp4
    """
    # Split the path to get individual components
    path_parts = input_path.split(os.sep)
    
    # Extract the video ID (assuming it's always the element before the last one in the path)
    video_id = path_parts[-2]
    
    # Construct the new path (assuming the structure you provided remains consistent)
    video_path = os.path.join(os.path.dirname(input_path), f"{video_id}.mp4")
    
    return video_path


def get_traj(traj_data_dic, args, cap_in, video_dir_path):
    """
    Constructs a trajectory from processed video data using information stored in various files like masks,
    metadata, and occlusion images. This function reads necessary data for each frame and compiles them into a 
    cohesive trajectory object.

    Args:
    traj_data_dic (dict): A dictionary mapping frame IDs to file paths for full masks, occlusion images, and metadata. # e.g. {'0000001': {'full_mask': '/../000001_full_mask_npy', 'occlusion': ..., 'metadata':...}, '00000002': {....}, ...} 
    args (Namespace): Contains settings and thresholds used to initialize animal instances.
    cap_in (cv2.VideoCapture): An OpenCV video capture object initialized to the video file.
    video_dir_path (str): The path to the directory containing the video and associated data.

    """
    traj = []
    for i, (frame_id, value) in enumerate(tqdm(traj_data_dic.items())):

        frame_id_int = int(frame_id)
        with open(value['metadata'], 'r') as file:
                metadata = json.load(file)
        frame_size = (metadata['full_w'], metadata['full_h'])
        cap_in.set(cv2.CAP_PROP_POS_FRAMES, frame_id_int)
        _, im = cap_in.read()
        im = pil_resize(im, (frame_size[0], frame_size[1]), mode=Image.BILINEAR) #im_crop


        
        # Read the image in grayscale mode to automatically handle the 3-channel repetition
        mask = cv2.imread(value['full_mask'], cv2.IMREAD_GRAYSCALE)

        # Convert the image back to the original 2D array with values between 0 and 1
        mask = mask / 255.0

        score = metadata['segm_score']
        label = metadata['label']

        box = (metadata['w0_BB'], metadata['h0_BB'], metadata['w1_BB'], metadata['h1_BB'])

        occlusion_im = cv2.imread(value['occlusion'])

        # Convert the image from BGR to RGB color space
        avg_depth_map = cv2.imread(os.path.join(os.path.dirname(video_dir_path), 'all_depth_maps', frame_id + '.png'), cv2.IMREAD_UNCHANGED)

        curr_animal = Animal(args, im, frame_id_int, score, mask, label, box, args.percent_outside_threshold, avg_depth_map, occlusion_im, metadata['occluded_proportion'])
        if i ==0:
            traj = Trajectory(curr_animal)
        else:
            traj.add_to_trajectory(curr_animal)
    return traj


def filenames_to_dic(filenames, dirpath):

    """
    Organizes a list of filenames into a dictionary, categorizing them by a common identifier derived from the filenames
    and storing paths to these files. This structure supports efficient file retrieval for processing based on
    the file's designated role (e.g., mask, metadata).

    Args:
    filenames (list of str): A list of filenames to be organized.
    dirpath (str): The directory path where the files are located, used to construct full file paths.

    Returns:
    dict: A dictionary where each key is a unique file index derived from the filenames, and each value is another
          dictionary mapping file types (e.g., 'full_mask', 'metadata') to their full paths.

    e.g. {'0000001': {'full_mask': '/../000001_full_mask_npy', 'mask': ..., 'metadata':...}, '00000002': {....}, ...} 
    """

    full_dic = {}

    for file in filenames:
        parts = file.split('_')
        f_index = parts[0]
        rest_of_filename = '_'.join(parts[1:])
        if rest_of_filename in file_types:
            if f_index not in full_dic:
                full_dic[f_index] = {}
            full_dic[f_index][rest_of_filename.split('.')[0]] = os.path.join(dirpath, file)

    sorted_dict = {key: full_dic[key] for key in sorted(full_dic)}
    return  sorted_dict   

def get_any_file(directory):
    # List all files and directories in the given directory
    files_and_dirs = os.listdir(directory)
    
    # Filter out directories, keep only files
    files = [f for f in files_and_dirs if os.path.isfile(os.path.join(directory, f))]
    
    # Select a random file (or any file) if there are any files present
    if files:
        file_path = os.path.join(directory, random.choice(files))
        return file_path
    else:
        return None  # Return None if there are no files

def process_vid(video_dir_path, gpu_queue, args): #video directory path, e.g. /...../all_Bb8nquNkruY_clips_after_1st_stage

    """
    Processes a single video directory by handling video flow computation and trajectory processing, 
    leveraging specific GPU resources allocated via a queue. This function is part of a larger video 
    processing pipeline that processes videos staged in a specified directory structure.

    Args:
    video_dir_path (str): The path to the directory containing video data to be processed, typically 
                          including initial processing results like segmented clips.
    gpu_queue (Queue): A multiprocessing queue used to manage GPU resources, ensuring that each processing 
                       task uses a designated GPU.
    args (Namespace): A namespace object containing configuration and operational parameters such as output sizes,
                      paths to models, and other necessary details.
    """

    # get an available GPU
    gpu_id = gpu_queue.get()
    try:
        # specify the GPU to run the script on
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cap_in = cv2.VideoCapture(get_video_path(video_dir_path))
        
        video_id = Video_ID()
        
        script_path = os.path.dirname(os.path.abspath(__file__))
        
        
        script_parent_path = os.path.abspath(os.path.join(script_path, '..'))
        
        sys.path.insert(0, os.path.join(script_parent_path,"externals/RAFT"))
        sys.path.insert(0, os.path.join(script_parent_path,"externals/RAFT/core"))
        
        
        # sys.path.append(os.path.join(script_parent_path,"externals/RAFT")) 
        # sys.path.append(os.path.join(script_parent_path,"externals/RAFT/core")) 
        sys.path.append(os.path.dirname(script_path))
        sys.path.append(os.path.dirname(script_parent_path))
        from track_utils.flow import FlowModel
        model_flow = FlowModel(os.path.join("externals","RAFT", "models", "raft-sintel.pth"), 'cuda')
        out_size = (int(args.out_size), int(args.out_size))
        for dirpath, dirnames, filenames in tqdm(os.walk(video_dir_path)):
            #e.g. dirpath = /...../all_Bb8nquNkruY_clips_after_1st_stage/00000
            #e.g. dirnames = the directoriess in 00000, it is empty
            #all the files in dirpath\
            if dirpath == video_dir_path:
                continue
           
            video_id.id = int(dirpath.split('/')[-1])
            out_dir = os.path.dirname(dirpath).replace('1st_stage', '2nd_stage') #e.g. /...../all_Bb8nquNkruY_clips_after_2nd_stage
            

            traj_data_dic = filenames_to_dic(filenames=filenames, dirpath=dirpath)
            frame_size = cv2.imread(get_any_file(os.path.join(os.path.dirname(os.path.dirname(dirpath)), 'all_depth_maps'))).shape
            traj = get_traj(traj_data_dic, args, cap_in, video_dir_path)
            save_trajectory_2nd_stage(args=args, out_dir=out_dir, video_id = video_id, full_h=frame_size[0], full_w=frame_size[1], traj=traj, model_flow=model_flow, out_size=out_size)
        
    finally:
        # release the GPU
        gpu_queue.put(gpu_id)

        
        
def process_all_videos(videos_dir_list, num_gpus, args):   
    """
    Manages the parallel processing of videos across multiple GPUs. This function sets up a multiprocessing
    environment to handle video processing tasks concurrently, distributing them across available GPUs.

    Args:
    videos_dir_list (list of str): A list of directory paths where each directory contains video data to be processed.
    num_gpus (int): The number of available GPUs to distribute the processing tasks.
    args (Namespace): A namespace object that contains all necessary configuration and parameters needed for video processing.
    """
    wait_to_preprocess = []
    for video in videos_dir_list:
        wait_to_preprocess.append(video)

    
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for i in range(num_gpus):
        gpu_queue.put(i)

    with mp.Pool(num_gpus) as p:
        results = [p.apply_async(process_vid, args=(video_path, gpu_queue, args)) for i, video_path in enumerate(wait_to_preprocess)] #video directory path, e.g. /...../all_Bb8nquNkruY_clips_after_SAM_occlusion
        # wait for all processes to finish
        [r.get() for r in results]


def main(args):
    """
    Identifies and processes directories containing video data based on specific naming conventions within a given base directory. 
    It filters directories that contain a specified substring, sorts them, and then processes them using defined GPU resources.
    """
    videos_dir_list = []
    # Setting base directory here
    base_directory = args.curr_base_dir
    # Loop through each directory and subdirectory in the base directory
    for dirpath, dirnames, filenames in os.walk(base_directory):
        for dirname in dirnames:
            # Construct the full path to the current subdirectory
            subdirectory_path = os.path.join(dirpath, dirname)
            # Check if the subdirectory name contains 'folder_name_ext'
            if '1st_stage' in dirname:
                # Append the subdirectory path to the list
                videos_dir_list.append(subdirectory_path)
    sorted_videos_dir_list = sorted(videos_dir_list)
    sorted_videos_dir_list =sorted_videos_dir_list[args.video_n_init:args.video_n_end]
    print("sorted_videos_dir_list: ", sorted_videos_dir_list)
    process_all_videos(videos_dir_list=sorted_videos_dir_list, num_gpus=args.num_gpus, args = args)

if __name__ == '__main__':

    parser = configargparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, is_config_file=True, help='Specify a config file path')  
    parser.add_argument('--video_n_init', type=int, default=0, help='the beggining of video ids to be processed')
    parser.add_argument('--video_n_end', type=int, default=1, help='the end of video ids to be processed')
    parser.add_argument('--curr_base_dir', type=str,  help='Specify the base directory path, where it contains folders for each video') 
    parser.add_argument('--folder_name_ext', type=str, help='Specify the folder extension you want to work with, e.g. SAM_occlusion')  
    parser.add_argument('--num_gpus', type=int, default=2, help='number of available gpus')
    parser.add_argument('--percent_outside_threshold', type=float)
    parser.add_argument('--out_size', type=int, default=256, help='')
    parser.add_argument('--ratio_crop_size', type=float) 
    parser.add_argument('--ignore_cumulative_flow', action='store_true', help='')   
    parser.add_argument('--save_motion', action='store_true', help='')
    parser.add_argument('--ignore_occlusion', action='store_true', help='') 
    parser.add_argument('--allowed_occlusion_perc', type=float, help=' above this threshold images will be thrown away')  
    parser.add_argument('--smooth_avg_len', type=int, default=5, help='the smoothing averaging segment length.') 
    parser.add_argument('--allowed_truncation_perc', type=float, help=' above this threshold images will be thrown away, this is for border trunction')   
    parser.add_argument('--add_crop_margin', type=float, help=' the margin to be added to the crop to make sure it includes all the object')

    parser.add_argument('--video_min_time', type=int, default=0, help='minimum number of frames in each traj')
    parser.add_argument('--video_max_time', type=int, default=200, help='maximum number of frames in each traj')

    args, _ = parser.parse_known_args()
    
    main(args)
