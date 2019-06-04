import os
import pdb
import glob 

"""
Script to create an input file containing the paths of images for YOLO's inference. 
The code: 
1. Accesses the subject data from multiwork 
2. Writes the appropriate image paths to an input file 
"""

def create_input_file(input_filename): 
  """Given a file name, create an input file in the data/ folder. The input file 
  contains a list of frame paths, separated by a new line.
  """
  # Path to exp12 data on Salk
  multiwork_path = '/marr/multiwork/experiment_12/included/'
  # List of subject directories for which we don't have size/distance variables
  list_sub_dirs_no_vars = ['__20161026_18625', '__20170322_18742', '__20170422_17919', '__20170928_19615', '__20170929_19954', '__20171003_18996', '__20171010_19544', '__20171012_19694', '__20171012_21015', '__20180126_19859', '__20180305_20510']

  # Write frame names to input file 
  with open(os.path.join('../data/', input_filename), 'w') as infile: 
    # Run YOLO on each frame for each of the above subjects
    for sub_dir in list_sub_dirs_no_vars: 
      list_child_frames = glob.glob(os.path.join(multiwork_path, sub_dir, 'cam07_frames_p/*'))
      infile.write('\n'.join(list_child_frames))
      infile.write('\n')


if __name__=='__main__': 
  create_input_file('test_input.txt')
