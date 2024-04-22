import configargparse
import os
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import shutil




def main(args):
    
    
    base_directory = args.base_dir
    out_directory = args.out_dir
    curr_id = 0
    full_dataset = {}
    # Loop through each directory and subdirectory in the base directory
    for dirpath, dirnames, filenames in os.walk(base_directory):
        for dirname in dirnames:
            # Construct the full path to the current subdirectory
            subdirectory_path = os.path.join(dirpath, dirname)
            # Check if the subdirectory name contains 'folder_name_ext'
            if '2nd_stage' in dirname:
                # Append the subdirectory path to the list
                for root, dirs, files in os.walk(subdirectory_path):
                    for name in dirs:
                        curr_path = os.path.join(root, name)
                        full_dataset[curr_id] = curr_path
                        curr_id += 1
    
    # Convert dictionary to a list of tuples to be able to shuffle and split them
    items = list(full_dataset.items())

    # Splitting the items into training and testing sets
    train_items, test_items = train_test_split(items, test_size= 1 - args.train_perc, random_state=42)

    # Convert lists back to dictionaries
    train_dict = dict(train_items)
    test_dict = dict(test_items)

    print(f"Training set size: {len(train_dict)}")
    print(f"Testing set size: {len(test_dict)}")

    # Directories for training and testing sets
    train_dir = os.path.join(out_directory, 'train')
    test_dir = os.path.join(out_directory, 'test')

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to copy directories to respective directories with new names based on ids
    def copy_directories_to_directory(directories_dict, destination_directory):
        for directory_id, directory_path in directories_dict.items():
            # Constructing new directory path using ID
            new_directory_path = os.path.join(destination_directory, '%07d'%directory_id )
            # Copying the entire directory tree
            shutil.copytree(directory_path, new_directory_path)

    # Copying training directories
    copy_directories_to_directory(train_dict, train_dir)

    # Copying testing directories
    copy_directories_to_directory(test_dict, test_dir)

    print("Directories have been copied to their respective directories.")



if __name__ == '__main__':

    parser = configargparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', type=str,  help='Specify the base directory path, where it contains folders for each video') 
    parser.add_argument('--train_perc', type=float) #0.8
    parser.add_argument('--out_dir', type=str, help='the output directory path') #/.../Final


    args, _ = parser.parse_known_args()
    
    main(args)
