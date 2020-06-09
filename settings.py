import torch

if torch.cuda.is_available():
    model_dicts = {
        'MFF_resnet': '/home/u2263506/pretrained_model/MFF/model_resnet',
        'MFF_senet': '/home/u2263506/pretrained_model/MFF/model_senet',
        'MFF_densenet': '/home/u2263506/pretrained_model/MFF/model_densenet_nodot',
    }
else:
    model_dicts = {
        'MFF_resnet': '/Users/youzunzhi/pro/EVA/source_code/Visualizing-CNNs-for-monocular-depth-estimation/pretrained_model/model_resnet',
        'MFF_senet': '/Users/youzunzhi/pro/EVA/source_code/Visualizing-CNNs-for-monocular-depth-estimation/pretrained_model/model_senet',
        'MFF_densenet': '/Users/youzunzhi/pro/EVA/source_code/Visualizing-CNNs-for-monocular-depth-estimation/pretrained_model/model_densenet',
    }


######### global settings  #########
GPU = torch.cuda.is_available()             # running on GPU is highly suggested
TEST_MODE = not torch.cuda.is_available()       # turning on the testmode means the code will run on a small nyu_dataset.
# TEST_MODE = True       # turning on the testmode means the code will run on a small nyu_dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
MODEL_NAME = 'MFF_resnet'                   # model arch: mff_resnet
MODEL_WEIGHTS_FILE = model_dicts[MODEL_NAME]
DATASET = 'nyu'                             # model trained on: nyu
CATAGORIES = ["rel"]                        # concept categories that are chosen to detect: sem|abs|rel
DATA_DIRECTORY = f'dataset/nyuv2/{CATAGORIES[0]}_csv'
IMG_SIZE = [228, 304]
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
OUTPUT_FOLDER = f"result/{MODEL_NAME}_{DATASET}_{CATAGORIES[0]}" # result will be stored in this folder

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden nyu_dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

if MODEL_NAME == 'MFF_resnet':
    # FEATURE_NAMES = ['MFF', 'D']
    FEATURE_NAMES = ['MFF']
else:

    raise NotImplementedError

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 2
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    # INDEX_FILE = 'index_sm.csv' # copy some lines(as you like) from file 'nyu_dataset/broden1_224/index.csv'.
    INDEX_FILE = f'dense_{CATAGORIES[0]}_index.csv'
    OUTPUT_FOLDER += "_debug"
else:
    WORKERS = 1
    BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = f'index.csv'
    if CATAGORIES[0] != 'sem':
        INDEX_FILE = f'dense_{CATAGORIES[0]}_index.csv'
        assert INDEX_FILE.find(CATAGORIES[0])!=-1
