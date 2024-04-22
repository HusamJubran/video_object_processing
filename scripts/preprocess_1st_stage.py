import os
import matplotlib.pyplot as plt
import cv2
import subprocess
import multiprocessing as mp
import configargparse 



class Video_Input:

    """
    A class for managing video input operations, including file handling, directory management, 
    and basic video properties extraction. It is tailored to handle specific videos, organizing them 
    into designated directories, copying and removing files as needed, and retrieving video statistics.

    Attributes:
    base_path (str): The base directory where the video files are stored.
    video_label (str): A label describing the video, used for categorization or identification.
    video_id (str): The identifier for the video, extracted from the video's filename.
    video_path (str): The full path to the specific directory for the video.
    link (str): A URL link constructed using the video_id, presumed to lead to a video resource.
    video_mp4_name (str): The name of the video file within its specific directory.
    num_of_frames (int): The total number of frames in the video, extracted via OpenCV.
    time_in_sec (float): The total duration of the video in seconds, calculated from the frame count.
    preprocess_flag (bool): Flag indicating whether preprocessing has been completed for the video.
    """

    def __init__(self, folder_path, video_full_name, video_label): #horse_new, specific video

        """
        Initializes an instance of the Video_Input class, setting up directories and managing video files.

        Args:
        folder_path (str): Path to the directory where video files are initially stored.
        video_full_name (str): The filename of the video, including its extension.
        video_label (str): A descriptive label for the video, used for organizational or processing purposes.
        """

        self.base_path = folder_path
        self.video_label = video_label

        self.video_id = video_full_name.split(".")[0]
        self.video_path = folder_path + self.video_id + "/" 
        print("self.video_path: ", self.video_path )
        self.link = "https://artgrid.io/clip/" + self.video_id
        
        if os.path.exists(self.video_path) == False:
            #there is no folder for the video, create one and remove the video from the folder. 

            os.makedirs(self.video_path, exist_ok=True) #create video folder
            os.system("cp " + folder_path + video_full_name + " " + self.video_path) #copy the video to the folder
            os.system("rm " + folder_path + video_full_name) #remove the video from the main folder
        
        files_in_video_dir = os.listdir(self.video_path)
        print("files_in_video_dir", files_in_video_dir)
        self.video_mp4_name = [i for i in files_in_video_dir if i.startswith(self.video_id)]
        print("self.video_mp4_name",self.video_mp4_name)
        if self.video_mp4_name != []:
            self.video_mp4_name = self.video_mp4_name[0]

            print("self.video_mp4_name", self.video_mp4_name)
            #Save statistics on the video 
            video = cv2.VideoCapture(self.video_path + self.video_mp4_name)
            self.num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.time_in_sec = round(self.num_of_frames/30, 2)
        else:
            self.video_mp4_name = ""

        #check if the video has already run with the preprocess: 
        self.preprocess_flag = self.preprocess_exist_new()

        print("self.preprocess_flag", self.preprocess_flag)


        self.number_of_output_videos = None
        self.total_time_of_output_videos = None

    def calculate_out_time(self):
        """
        Calculates the total output time of processed video clips and updates the instance with the total count of output videos.

        """
        post_vid = os.listdir(self.video_path + "/all_" + self.video_id +"_clips_after")
        self.number_of_output_videos = len(post_vid)
        frames = 0 
        for post in post_vid:
            files_in_vid = os.listdir(self.video_path + "/all_" + self.video_id +"_clips_after/" + post)
            num_of_files_in_vid = len(files_in_vid) / 3 #Mask,BB,Frame
            frames = frames + num_of_files_in_vid

        self.total_time_of_output_videos = round(frames/30, 2)


    def preprocess_exist_new(self):
        """
        Checks if preprocessing has already been completed for the video by verifying the existence of a directory 
        that would contain the post-processed clips. If this directory exists and is not empty, it further updates 
        the total time of the output videos.

        Returns:
        bool: True if the preprocessing results exist and the directory contains files, otherwise False.

        """
        files_in_video_dir = os.listdir(self.video_path)
        if os.path.exists(self.video_path + "/all_" + self.video_id +"_clips_after") == True and len(os.listdir(self.video_path + "/all_" + self.video_id +"_clips_after")) > 0:
                self.calculate_out_time()
                return True
        return False


############################################################################################################

def pre_process_vid(video, gpu_queue, path_to_images_folder, config_path):

    """
    Handles the preprocessing of a video by running an external script (preprocess_1st_stage_per_video.py) that performs operations
    such as object detection, segmentation, and frame extraction. The function assigns a GPU for 
    the task, prepares the command to execute the script, and manages GPU resource allocation.

    Args:
    video (Video_Input): An instance of Video_Input containing video metadata and path information.
    gpu_queue (queue.Queue): A queue for managing GPU resource IDs to ensure proper allocation and 
                             synchronization across processes.
    path_to_images_folder (str): The path to the folder where images from the video are to be stored.
    config_path (str): The path to the configuration file needed for the preprocessing script.


    Returns:
    None: This function does not return a value but ensures the external script is executed and the GPU
          resource is managed correctly.
    """

    # get an available GPU
    gpu_id = gpu_queue.get()
    try:
        # specify the GPU to run the script on
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


        video_file = video.video_path + video.video_mp4_name
        valid_portion = 1.0
        
        if video_file[-4:] == ".mp4" or video_file[-4:] == ".mov" or video_file[-4:] == ".MOV":
            cap_in = cv2.VideoCapture(video_file)
            total_frames_num = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT) * valid_portion) #TODO: it is already calculated
        else:
            cap_in = None
            print("video_file", video_file)
            image_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and (f.lower().endswith(('.png')) or f.lower().endswith(('.jpg')))]

            total_frames_num = len(image_files)
        
        print("number of frames", total_frames_num)
        print("Detecting " + video.video_label + "s......")

        if total_frames_num != 0:

            # Get the directory of the current script
            current_script_dir = os.path.dirname(os.path.abspath(__file__))

            # Define the relative path to the project root from the current script
            # For example, if the script is in some subdirectory of the project
            relative_path_to_project_root = os.path.join(current_script_dir, '..')

            print("relative_path_to_project_root : ", relative_path_to_project_root )


            cmd_run = 'python -m scripts.preprocess_1st_stage_per_video '+\
                    '--config  ' + config_path+' '+\
                    '--video_path '+video_file+' '+\
                    '--out_dir '+ os.path.join(video.video_path, "all_"+video.video_id+"_clips_after") +\
                    ' --total_frame_num ' + str(total_frames_num) + " " +\
                    '--target_object '+ video.video_label + " " +\
                    '--path_to_images_folder ' + path_to_images_folder
            subprocess.run(cmd_run, shell=True, cwd=relative_path_to_project_root)  
        else:
            print("there are 0 frames in video " + video.video_id)
        
    finally:
        # release the GPU
        gpu_queue.put(gpu_id)
    

def preprocess_videos(videos_list, num_gpus, path_to_images_folder, config_path):  

    """
    Preprocesses a list of videos in parallel, distributing the task across multiple GPUs. It uses asynchronous 
    processing to manage video preprocessing on available GPUs to enhance performance and reduce processing time.

    Args:
    videos_list (list of Video_Input): A list of Video_Input instances, each containing metadata and paths 
                                       for videos to be processed.
    num_gpus (int): The number of GPUs available for processing.
    path_to_images_folder (str): The path where images generated from video processing should be stored.
    config_path (str): The path to the configuration file necessary for video processing scripts.

    """


    wait_to_preprocess = []
    for video in videos_list:
        wait_to_preprocess.append(video)

    print("wait_to_preprocess", len(wait_to_preprocess))
    
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    
    for i in range(num_gpus):
        gpu_queue.put(i)

    with mp.Pool(num_gpus) as p:
        results = [p.apply_async(pre_process_vid, args=(video_path, gpu_queue, path_to_images_folder, config_path)) for i, video_path in enumerate(wait_to_preprocess)] #it is video not video path
        
        # wait for all processes to finish
        [r.get() for r in results]


def check_video_label(animals, folder_name):
    """
    Checks if any specified animal names exist within a given folder name and returns the first match.

    Args:
    animals (list of str): A list of animal names to check against the folder name.
    folder_name (str): The name of the folder to search for animal names.

    Returns:
    str: The name of the first animal found in the folder name, if any; otherwise, None.
    """
    animal_name = None
    for animal in animals:
        if animal in folder_name:
            animal_name = animal
            return animal_name
        
    return None


def create_videos_list(base_path, folder, num_gpus, config_path, video_n_init, video_n_end, check_length_dir): 
    """
    Prepares and initiates preprocessing for a list of videos located within a specified directory. 
    The function determines the type of animal in the videos based on the folder name, sets up directories 
    for image data, and manages the preprocessing tasks on available GPUs.

    Args:
    base_path (str): The base directory where video data are stored. #"/.../animals_dataset_v0/"  
    folder (str): Specific subdirectory from which videos are to be processed. #"horse_new"
    video (str): Specific video file name; if provided, only this video is processed.
    num_gpus (int): Number of available GPUs for processing.
    config_path (str): Configuration path for the preprocessing script.
    video_n_init (int): Index to start processing from within the filtered list of videos.
    video_n_end (int): Index to end processing in the filtered list of videos, exclusive.
    """
    coco_target_object = ['dog', 'cat', 'elephant', 'bear', 'zebra', 'giraffe', 'cow', 'sheep', 'horse'] 
    videos_list = []
    video_label = check_video_label(coco_target_object, folder)
    
    path_to_images_folder = os.path.join(base_path, folder, "images_data")
    print("path_to_images_folder", path_to_images_folder)
    print("config_path", config_path)
    if os.path.exists(path_to_images_folder) == False:
        os.makedirs(path_to_images_folder, exist_ok=True) #create video folder
    
    print("videos_in_folder: ", sorted(os.listdir(base_path + folder)))
       
    if video_label == None:
        print("ERROR in finding animal")
    
    else:
        videos_in_folder = sorted(os.listdir(base_path + folder))
        
        for video_name in videos_in_folder:
            print("video_name.split", video_name.split("/")[-1])
            if video_name[0] == "." or video_name.endswith(".csv") or video_name.split("/")[-1] == "images_data":
                continue
            # elif check_length_dir == True and len(os.listdir(os.path.join(base_path, folder, video_name))) > 1:
            #     continue # this means the video folder is processed, if the pre-process fails, remember to delete every folders besides the original video
            else: 
                new_video = Video_Input(base_path + folder + "/" ,video_name , video_label)
                videos_list.append(new_video)
                
    videos_list = videos_list[video_n_init:video_n_end]
    print("videos_list[0].video_path: ", videos_list[0].video_path)
    preprocess_videos(videos_list, num_gpus, path_to_images_folder,config_path)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='') 
    parser.add_argument('--base_path', type=str, help='') #"/.../animals_dataset_v0/" 
    parser.add_argument('--curr_folder', type=str, help='') #"horse_new"
    parser.add_argument('--num_gpus', type=int)
    parser.add_argument('--check_length_dir', action='store_true', help='')  
    parser.add_argument('--config', type=str, is_config_file=True, help='Specify a config file path')
    parser.add_argument('--video_n_init', type=int)
    parser.add_argument('--video_n_end', type=int)

    args, _ = parser.parse_known_args()

    
    create_videos_list(args.base_path, args.curr_folder, args.num_gpus, args.config, args.video_n_init, args.video_n_end, args.check_length_dir)