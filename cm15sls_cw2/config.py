"""
Parameters. You will need to customize the directories to match your
environment.

"""


##################################################################
# 
#         		DIRECTORIES
# 
################################################################### 


# path to caption data
CAPTION_DIR = "C:\\Users\\Sarah Smith\\Google Drive\\University\\Leeds\\CDT\\Year 1\\Semester 2\\COMP5623M - Artificial Intelligence\\Coursework\\Coursework 2\\flickr8k-custom\\captions\\"
# path to images
IMAGE_DIR = "C:\\Users\\Sarah Smith\\Google Drive\\University\\Leeds\\CDT\\Year 1\\Semester 2\\COMP5623M - Artificial Intelligence\\Coursework\\Coursework 2\\Flicker8k_Dataset\\"

# token file names to read from
TOKEN_FILE_TRAIN = CAPTION_DIR + "Flickr8k_train.token.txt"
TOKEN_FILE_TEST = CAPTION_DIR + "Flickr8k_test.token.txt"


##################################################################
# 
#         TRAINING and VOCAB PARAMS (defaults - you may change)
# 
################################################################### 

MIN_FREQUENCY = 3

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LR = 0.001
NUM_EPOCHS = 5
LOG_STEP = 10


