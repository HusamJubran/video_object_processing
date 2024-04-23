import configargparse
import os
import subprocess






def main(args):
    
    
    cmd_run = 'python externals/extract_dino/extract.py ' +\
                  '-c ' + args.config_DINO +\
                  ' --test_root ' + os.path.join(args.curr_base_dir, 'test') +\
                  ' --train_root ' + os.path.join(args.curr_base_dir, 'train') +\
                  ' --results_info_root ' + args.curr_base_dir
            

    subprocess.run(cmd_run, shell=True)  

if __name__ == '__main__':

    parser = configargparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, is_config_file=True, help='Specify a config file path')  
    parser.add_argument( '--config_DINO', type=str, is_config_file=True, help='Specify a config file path')  
    parser.add_argument('--curr_base_dir', type=str,  help='Specify the base directory path, where it contains test and train folders') #/.../Final

    args, _ = parser.parse_known_args()
    
    main(args)
