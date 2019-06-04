import os
import csv
import pdb
import glob 
from collections import defaultdict

"""
Script to break the big prediction file down into one for each subject
"""

def write_file(file_path, file_contents): 
  """
  """ 
  with open(file_path, 'w') as outfile: 
    outfile.write(file_contents) 
  return None


def get_subdir_from_frame_path(frame_path): 
  return os.path.basename(os.path.dirname(os.path.dirname(frame_path)))


def create_new_pred_file_for_sub(list_subs_pred_file_created, preds): 
  """Create a new prediction file for the appropriate subject. 
  -------
  Arguments
  ---------
  list_subs_pred_file_created: List of subjects which have been encountered so far
  preds: Dictionary with subject directory as key and predictions as value 
  --------
  Return
  --------
  list_subs_pred_file_created: Updated list of subjects includes the new subject 
  --------
  Output 
  --------
  'sub_dir_pred_file.txt': Prediction file for subject. 
  """
  pdb.set_trace()
  # Find subject whose predictions are in preds but prediction file isn't created
  cur_sub_dir = set(preds.keys()).difference(set(list_subs_pred_file_created)).pop()
  
  # Write predictions from current subject into file
  filename = 'pred_file'+cur_sub_dir+'.txt' 
  write_file(filename, preds[cur_sub_dir])  
  print('Wrote predictions to {}'.format(filename)) 

  # Update the list of subjects whose predictions are written into file 
  list_subs_pred_file_created.append(cur_sub_dir)
  
  return list_subs_pred_file_created
  

def breakdown(results_filename): 
  """
  return predictions of the format: 
  {'__20180932_233' (subject dir): 'pred1 \n pred2 \n pred3 ...', 'next_sub_dir': predictions <str>, ...}
  """
  # List of subject directories for subjects whose variables we'd like to compute
  list_sub_dirs_no_vars = ['__20161026_18625', '__20170322_18742', '__20170422_17919', '__20170928_19615', '__20170929_19954', '__20171003_18996', '__20171010_19544', '__20171012_19694', '__20171012_21015', '__20180126_19859', '__20180305_20510']

  # Store each prediction
  preds = {}

  # List of subjects for whom a prediction file is created
  list_subs_pred_file_created = []

  # Read prediction file containing predictions for all frames, subjects
  with open(results_filename, newline='') as csvfile: 
    reader_obj = csv.reader(csvfile, delimiter=' ')	

    # Iterate over each line from the prediction file
    count_entries = 0
    for entries in reader_obj: 
	
      # Parse prediction to find subject 
      sub_dir = get_subdir_from_frame_path(entries[0])

      # If subject is new then do a few things...
      if sub_dir not in preds: 
        # Initialize
        preds[sub_dir] = '' 
        # New subject being encountered marks the end of predictions from old subject. So store them. 
        if len(list_subs_pred_file_created) > 0: 
          list_subs_pred_file_created = create_new_pred_file_for_sub(list_subs_pred_file_created, preds)
      
      # Add new prediction to existing predictions
      preds[sub_dir] += ' '.join(entries)

      # Log process on console
      count_entries += 1
      if count_entries % 1000 == 0: 
        print('Processed {} entries.'.format(count_entries))


  # Display list of subjects encountered in prediction file to verify that 
  # subjects don't repeat -- if they do, then the above program logic is wrong. 
  print('List of subjects encountered in prediction file: {}'.format(list_subs_pred_file_created))

  return None


if __name__=='__main__': 
  breakdown('test_preds.txt')
