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
TEST_MODE = not torch.cuda.is_available()       # turning on the testmode means the code will run on a small dataset.
CLEAN = True                                # set to "True" if you want to clean the temporary large files after generating result
MODEL_NAME = 'MFF_resnet'                   # model arch: mff_resnet
MODEL_WEIGHTS_FILE = model_dicts[MODEL_NAME]
DATASET = 'nyu'                             # model trained on: nyu
DATA_DIRECTORY = 'dataset/nyuv2'
IMG_SIZE = [228, 304]
QUANTILE = 0.005                            # the threshold used for activation
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score (in HTML file)
TOPN = 10                                   # to show top N image with highest activation for each unit
PARALLEL = 1                                # how many process is used for tallying (Experiments show that 1 is the fastest)
CATAGORIES = ["sem"] # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = f"result/{MODEL_NAME}_{DATASET}" # result will be stored in this folder

########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own

# if MODEL != 'alexnet':
#     DATA_DIRECTORY = 'dataset/broden1_224'
#     IMG_SIZE = 224
# else:
#     DATA_DIRECTORY = 'dataset/broden1_227'
#     IMG_SIZE = 227
#
# if DATASET == 'places365':
#     NUM_CLASSES = 365
# elif DATASET == 'imagenet':
#     NUM_CLASSES = 1000
# if MODEL == 'resnet18':
#     FEATURE_NAMES = ['layer4']
#     if DATASET == 'places365':
#         MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
#         MODEL_PARALLEL = True
#     elif DATASET == 'imagenet':
#         MODEL_FILE = None
#         MODEL_PARALLEL = False
# elif MODEL == 'densenet161':
#     FEATURE_NAMES = ['features']
#     if DATASET == 'places365':
#         MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
#         MODEL_PARALLEL = False
# elif MODEL == 'resnet50':
#     FEATURE_NAMES = ['layer4']
#     if DATASET == 'places365':
#         MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
#         MODEL_PARALLEL = False

if MODEL_NAME == 'MFF_resnet':
    FEATURE_NAMES = ['D']
else:
    raise NotImplementedError

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 2
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv' # copy some lines(as you like) from file 'dataset/broden1_224/index.csv'.
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 1
    BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index.csv'
