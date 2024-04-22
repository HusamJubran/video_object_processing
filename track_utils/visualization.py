from ipywidgets import interactive
import ipywidgets as widgets
from IPython.display import clear_output
import os
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import torch
import matplotlib.cm
import plotly.express as px
import math

HORSE_VIDEOS_PATH = '/viscam/projects/video_animals/husam/preprocessing/data/horse_videos'
ANIMALS = {'horse': HORSE_VIDEOS_PATH, 'cat': 'temp'}
# cap_in = cv2.VideoCapture('/viscam/projects/video_animals/husam/preprocessing/data/horse_videos/Bb8nquNkruY/Bb8nquNkruY.mp4')



def pil_resize(im, size, mode=Image.BILINEAR):
    im = Image.fromarray(im)
    return np.array(im.resize(size, mode))

def get_dirs(driectory):   #e.g. /viscam/.../horse_videos

    # Get all items in the directory
    all_items = os.listdir(driectory)

    # Filter out items that are not directories
    directory_names = [item for item in all_items if os.path.isdir(os.path.join(driectory, item))]

    return directory_names


def get_random_color():
    color = np.random.choice(range(256), size=3)
    return (int(color[0]), int(color[1]), int(color[2])) 



def detect_borders_binary_mask(binary_mask, pool_size, base_dilation_kernel_size):
    # Ensure binary_mask is uint8 (values are 0 or 255)
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    # Apply average pooling
    pooled_image = cv2.blur(binary_mask_uint8, (pool_size, pool_size))

    # Compute the edge map
    edge_map = cv2.absdiff(binary_mask_uint8, pooled_image)

    # Threshold the edge map to binary
    _, edge_map = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)

    # Normalize edge_map to have values of 1 and 0
    edge_map_normalized = (edge_map / 255).astype(np.uint8)

    # Calculate the masked object size relative to the total image area
    object_area = np.sum(binary_mask) / np.product(binary_mask.shape)
    print("object_area: ", object_area)
    # Adjust dilation_kernel_size based on the object size
    # adjusted_dilation_kernel_size = max(1, int(base_dilation_kernel_size * object_area))
    adjusted_dilation_kernel_size = max(1, int(base_dilation_kernel_size * np.sqrt(object_area)))
    print("adjusted_dilation_kernel_size: ", adjusted_dilation_kernel_size)

    adjusted_dilation_kernel_size = base_dilation_kernel_size #------------------------------------------------------------------------------------
    # Apply dilation to thicken the edges
    kernel = np.ones((adjusted_dilation_kernel_size, adjusted_dilation_kernel_size), np.uint8)
    edge_map_dilated = cv2.dilate(edge_map_normalized, kernel, iterations=1)
    # edge_map_dilated = edge_map_dilated* (1-binary_mask_uint8)  #-------------------------------------------------------
    return binary_mask_uint8, edge_map_dilated

def shrink_mask(mask, pixels=1):
    mask = torch.FloatTensor(mask)[None, None, :, :]
    kernel_size = pixels*2+1
    new_mask = torch.nn.functional.avg_pool2d(mask, kernel_size, stride=1, padding=pixels)[0,0]
    new_mask = (new_mask > 0.9999).float()
    return new_mask.numpy()


def apply_mask_and_fill(image, mask):
    """
    Applies a mask to an image and fills in the zeroed values with the nearest non-zero pixel value.
    """
    # Apply the mask
    masked_image = image * mask
    
    # Compute the distance transform
    distances, (indices_x, indices_y) = distance_transform_edt(mask == 0, return_indices=True)
    
    # Prepare an empty image to fill
    filled_image = masked_image[indices_x, indices_y]
    
    
    return filled_image, masked_image


# Updated colorize function that accepts global vmin and vmax
def colorize_infer(value, cmap='magma_r', vmin=None, vmax=None):
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # normalize to vmin..vmax
    else:
        value = value * 0.  # Avoid division by zero

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # Apply colormap

    value = value[..., :3]  # Drop the alpha channel
    return value


def conver_occ_to_rgb(border_depth, border, high_threshold, low_threshold ):
    mean_depth_difference = np.mean(border_depth)
    std_depth_difference = np.std(border_depth)
    # threshold = mean_depth_difference + 0.29*std_depth_difference
    # threshold = 0.1555

    border_indices = np.where(border == 1)

    # Extract the values from border_depth at the border pixel indices
    border_depth_values = border_depth[border_indices]

    # Calculate the mean and standard deviation of the values
    mean_value = np.mean(border_depth_values)
    std_dev_value = np.std(border_depth_values)

    print("Mean value of border depth:", mean_value)
    print("Standard deviation of border depth:", std_dev_value)
    
    print("calc threshold: ", mean_depth_difference + std_depth_difference)

    # Initialize an RGB image with black pixels
    rgb_image = np.zeros((*border_depth.shape, 3))

    total_border_size = np.sum(border)
    occluded_counter = 0
    not_occluded_counter = 0
    # Set pixel colors based on the mask and the values in array_to_display
    for i in range(border_depth.shape[0]):
        for j in range(border_depth.shape[1]):
            if border[i, j] == 1:
                if border_depth[i, j] > high_threshold:
                    rgb_image[i, j] = [0, 1, 0]  # Green for values above the threshold
                    not_occluded_counter += 1
                elif  border_depth[i, j] < high_threshold and  border_depth[i, j] > low_threshold:
                      rgb_image[i, j] = [0, 0, 1]  # blue
                else:
                    rgb_image[i, j] = [1, 0, 0]  # Red otherwise
                    occluded_counter += 1
                    

    print("occluded_counter: ", occluded_counter)
    print("not_occluded_counter: ", not_occluded_counter)
    print("total_border_size: ", total_border_size)
    print("occluded_counter/not_occluded_counter: ", occluded_counter/not_occluded_counter)

    return rgb_image



def find_global_min_max(*arrays):
    # Concatenate all arrays into a single array
    all_values = np.concatenate([arr.ravel() for arr in arrays])
    # Calculate the global minimum and maximum values
    global_vmin = all_values.min()
    global_vmax = all_values.max()
    return global_vmin, global_vmax


def interactive_calc_occlusion():
    
    
    widget_depth_map_path = widgets.Text(
        value='',  # Initial value is an empty string
        placeholder='Enter folder path',
        description='Depth Map Path:',
        disabled=False
    )

    widget_mask_path_path = widgets.Text(
        value='',  # Initial value is an empty string
        placeholder='Enter folder path',
        description='Mask Path:',
        disabled=False
    )

    
    
    widget_border_pool_size = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50000,
            step=1,
            description='Border Pool Size: ',
            disabled=False
        )
    

    widget_border_dilation_kernel_size = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50000,
            step=1,
            description='Border Dilation Size: ',
            disabled=False
        )
    
    widget_object_shrink_pxls = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50000,
            step=1,
            description='Object Shrink Pixels: ',
            disabled=False
        )
    
    widget_background_shrink_pxls = widgets.BoundedIntText(
            value=5,
            min=1,
            max=50000,
            step=1,
            description='Background Shrink Pixels: ',
            disabled=False
        )
    
    widget_high_threshold = widgets.BoundedFloatText(
            value=0.05,
            min=-50000,
            max=50000,
            step=1,
            description='Occlusion High Threshold: ',
            disabled=False
        )
    
    widget_low_threshold = widgets.BoundedFloatText(
            value=-0.07,
            min=-50000,
            max=50000,
            step=1,
            description='Occlusion Low Threshold: ',
            disabled=False
        )
    

    interactive_plot = interactive(calc_occlusion, depth_map_path = widget_depth_map_path, mask_path = widget_mask_path_path, border_pool_size = widget_border_pool_size, 
                                   border_dilation_kernel_size = widget_border_dilation_kernel_size, object_shrink_pxls = widget_object_shrink_pxls, 
                                   background_shrink_pxls = widget_background_shrink_pxls, high_threshold = widget_high_threshold, low_threshold= widget_low_threshold)

    output = interactive_plot.children[-1]
    output.layout.height = '400px'
    display(interactive_plot)

def calc_occlusion(depth_map_path, mask_path, border_pool_size, border_dilation_kernel_size, object_shrink_pxls, background_shrink_pxls, high_threshold, low_threshold):


    depth_map_path = '/viscam/projects/video_animals/husam/preprocessing/data/horse_videos/Bb8nquNkruY/all_depth_maps/0000001.npy'
    mask_path = '/viscam/projects/video_animals/husam/preprocessing/data/horse_videos/Bb8nquNkruY/all_Bb8nquNkruY_clips_after_SAM_occlusion/00000/0000001_full_mask.npy'

    
    depth_map_path = '/viscam/projects/video_animals/husam/preprocessing/data/horse_videos/SOXtcMUdCZk/all_depth_maps/0000022.npy'
    mask_path = '/viscam/projects/video_animals/husam/preprocessing/data/horse_videos/SOXtcMUdCZk/all_SOXtcMUdCZk_clips_after_SAM_occlusion/00000/0000022_full_mask.npy'


    depth_map = np.load(depth_map_path)
    mask = np.load(mask_path)

    object_area = np.sum(mask)

    object_area_sqrt = math.sqrt(object_area)
    print("object_area_sqrt: ", object_area_sqrt)
    border_dilation_kernel_size = object_shrink_pxls = background_shrink_pxls = round(object_area_sqrt/85)
    print("thikcness: ",  round(object_area_sqrt/85))
    binary_mask, border = detect_borders_binary_mask(mask, border_pool_size, border_dilation_kernel_size)

    # Apply the mask and fill the image
    shrinked_object = shrink_mask(mask,object_shrink_pxls)
    object_filled_depth_map ,object_masked_depth_map= apply_mask_and_fill(depth_map, shrinked_object)

    shrinked_background = shrink_mask(1 - mask,background_shrink_pxls)
    background_filled_depth_map, background_masked_depth_map = apply_mask_and_fill(depth_map, shrinked_background)

    object_border_depth = border * object_filled_depth_map
    background_border_depth = border * background_filled_depth_map
    border_depth_diff =  background_border_depth - object_border_depth




    global_vmin, global_vmax = find_global_min_max(object_border_depth, background_border_depth)

    # Apply the colorize function to each depth map
    # global_vmin = object_border_depth.min()
    # # global_vmax = np.percentile(object_border_depth, 95)
    # global_vmax = object_border_depth.max()
    colored_map1 = colorize_infer(object_border_depth, vmin=global_vmin, vmax=global_vmax)
    print(np.unique(object_border_depth))
    # global_vmin = background_border_depth.min()
    # global_vmax = np.percentile(background_border_depth, 95)
    colored_map2 = colorize_infer(background_border_depth, vmin=global_vmin, vmax=global_vmax)
    print(np.unique(background_border_depth))

    global_vmin, global_vmax = find_global_min_max(border_depth_diff)
    colored_map3 = colorize_infer(border_depth_diff, vmin=global_vmin, vmax=global_vmax)
    print(np.unique(border_depth_diff))

    # Now plotting the colorized depth maps
    fig, axs = plt.subplots(5, 2, figsize=(10, 20))

    axs[0][0].imshow(colored_map1)
    axs[0][0].set_title("object_border_depth")
    axs[0][0].axis('off')  # Hide the axis

    axs[0][1].imshow(colored_map2)
    axs[0][1].set_title("background_border_depth")
    axs[0][1].axis('off')

    axs[1][0].imshow(colored_map3)
    axs[1][0].set_title("border_depth_diff")
    axs[1][0].axis('off')

    

    global_vmin, global_vmax = find_global_min_max(object_filled_depth_map, background_filled_depth_map)

    
    colored_map = colorize_infer(object_filled_depth_map, vmin=global_vmin, vmax=global_vmax)
    axs[2][0].imshow(colored_map)
    axs[2][0].set_title("object depth fill")
    axs[2][0].axis('off')  # Hide the axis

    colored_map = colorize_infer(background_filled_depth_map, vmin=global_vmin, vmax=global_vmax)
    axs[2][1].imshow(colored_map)
    axs[2][1].set_title("background depth fill")
    axs[2][1].axis('off')  # Hide the axis


    axs[1][1].imshow(conver_occ_to_rgb(border_depth=border_depth_diff, border=border, high_threshold=high_threshold, low_threshold= low_threshold ))
    axs[1][1].set_title("Occlusion")
    axs[1][1].axis('off')  # Hide the axis


    axs[4][0].imshow(border)
    axs[4][0].set_title("object_border")
    axs[4][0].axis('off')  # Hide the axis


    global_vmin, global_vmax = find_global_min_max(depth_map)
    
    colored_map = colorize_infer(depth_map, vmin=global_vmin, vmax=global_vmax)

    axs[4][1].imshow(colored_map)
    axs[4][1].set_title("depth map")
    axs[4][1].axis('off')  # Hide the axis

    axs[3][0].imshow(shrinked_object)
    axs[3][0].set_title("shrinked_object")
    axs[3][0].axis('off')  # Hide the axis

    axs[3][1].imshow(shrinked_background)
    axs[3][1].set_title("shrinked_background")
    axs[3][1].axis('off')  # Hide the axis

    

    plt.tight_layout()
    plt.show()


    fig = px.imshow(colored_map)
    fig.update_traces(hoverinfo="x+y+z", hovertemplate="Pixel (x, y): (%{x}, %{y})<br>Value: %{z}")
    fig.show()



def interactive_video_segmentation():
    
    
    widget_animal = widgets.Dropdown(
        options=list(ANIMALS.keys()),
        value="horse",
        description='Animal: ',
        disabled=False,
    )

    
    
    widget_box_videos_list = widgets.Dropdown(
        options=["0000"],
        value="0000",
        description='Videos: ',
        disabled=False,
    )

    def update_animal_videos_list(change):
        new_animal = change['new']
        widget_box_videos_list.options = get_dirs(ANIMALS[new_animal])

    widget_animal.observe(update_animal_videos_list, names='value')

    # widget_box_videos_list = widgets.Dropdown(
    #     options=["Bb8nquNkruY"],
    #     value="Bb8nquNkruY",
    #     description='Videos: ',
    #     disabled=False,
    # )


    widget_box_frame_num = widgets.BoundedIntText(
            value=0,
            min=0,
            max=50000,
            step=1,
            description='Frame id: ',
            disabled=False
        )

    
    
    # widget_box_frame_num = widgets.Dropdown(
    #     options=list(range(500)),
    #     value = 0,
    #     description='Frame id: ',
    #     disabled=False,
    # )
    

    
    

    #interactive_plot = interactive(plot_pca, epoch_n=widgets.IntSlider(min=snapshot_freq, max=last_epoch_n, step=snapshot_freq, value=0), expdir = expdir)
    interactive_plot = interactive(video_segmentation,animal = widget_animal, video_id = widget_box_videos_list, frame_num = widget_box_frame_num)

    output = interactive_plot.children[-1]
    output.layout.height = '400px'
    display(interactive_plot)


def get_distinct_colors(n):
    hsv_colors = [(int(i * 255 / n), 255, 255) for i in range(n)]  # Full saturation and value
    rgb_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] for hsv in hsv_colors]
    return rgb_colors


def video_segmentation(animal, video_id, frame_num):
    #e.g. video_id = A6784FAASF, animal = 'horse'
    curr_video_dir = os.path.join(ANIMALS[animal], video_id) #e.g. /viscam/.../A6784FAASF
    full_video_path = os.path.join(curr_video_dir, video_id + '.mp4') #e.g. /viscam/.../A6784FAASF/A6784FAASF.mp4
    curr_1st_stage = os.path.join(curr_video_dir, "all_"+video_id+"_clips_after" + "_" + '1st_stage') #/viscam/.../A6784FAASF/all_clips_after...
    frame_size = cv2.imread(os.path.join(curr_video_dir, 'all_depth_maps', '0000000.png')).shape
    print("curr_1st_stage: ", curr_1st_stage)
    trajectories = get_dirs(curr_1st_stage) #[00000, 000001,...]
    print("trajectories: ", trajectories)
    # colors = [get_random_color() for traj in trajectories]
    colors = [(int(traj) * 50 % 255, int(traj) * 70 % 255, int(traj) * 90 % 255)  for traj in trajectories]
    # colors = [get_distinct_colors(int(traj)) for traj in trajectories]


    cap_in = cv2.VideoCapture(full_video_path)
    cap_in.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, im = cap_in.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = pil_resize(im, (frame_size[1], frame_size[0]), mode=Image.BILINEAR) #im_crop

    for color, traj in zip(colors, trajectories):
        metadata_file_path = os.path.join(curr_1st_stage, traj, '%07d' %(frame_num) + '_metadata.json')
        metadata_file_path_2nd = os.path.join(curr_1st_stage, traj, '%07d' %(frame_num) + '_metadata.json').replace('1st_stage', '2nd_stage')
        if os.path.exists(metadata_file_path):
            # The file exists, so open and load it
            print("metadata_file_path: ", metadata_file_path)
            with open(metadata_file_path, 'r') as file:
                metadata = json.load(file)
            
            with open(metadata_file_path_2nd, 'r') as file:
                metadata_2nd = json.load(file)
            
            im = cv2.rectangle(im, (int(metadata['w0_BB']), int(metadata['h0_BB'])), (int(metadata['w1_BB']), int(metadata['h1_BB'])), color=color, thickness=10)
            im = cv2.rectangle(im, (int(metadata_2nd['xmin']), int(metadata_2nd['ymin'])), (int(metadata_2nd['xmin'] + metadata_2nd['w']), int(metadata_2nd['ymin']) + metadata_2nd['h']), color=(0, 0, 255), thickness=10)

        occlusion_file_path = os.path.join(curr_1st_stage, traj, '%07d' %(frame_num) + '_occlusion.png')
        if os.path.exists(occlusion_file_path):
            # The file exists, so open and load it
            occlusion_mask = cv2.imread(occlusion_file_path)
            occlusion_mask_rgb = cv2.cvtColor(occlusion_mask, cv2.COLOR_BGR2RGB)
            mask_indices = np.where((occlusion_mask_rgb != [0, 0, 0]))
            im[mask_indices] = occlusion_mask_rgb[mask_indices]




    plt.imshow(im)
    plt.title(f'Frame {frame_num} ')
    plt.axis('off')
    plt.show()

    print("hi")





