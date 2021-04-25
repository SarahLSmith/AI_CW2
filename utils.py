import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import string
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

from vocabulary import Vocabulary
from config import *



def read_lines(filepath):
    """ Open the ground truth captions into memory, line by line. 
    Args:
        filepath (str): the complete path to the tokens txt file
    """
    file = open(filepath, 'r')
    lines = []

    while True: 
        # Get next line from file 
        line = file.readline() 
        if not line: 
            break
        lines.append(line.strip())
    file.close()
    return lines

def get_image_ref_captions_dict(image_id_lst, ref_captions):
    image_ref_dict = dict()
    for idx, image in enumerate(image_id_lst):
        if image not in image_ref_dict:
            image_ref_dict[image] = list()
        image_ref_dict[image].append(ref_captions[idx])
    
    clean_ref_captions(image_ref_dict)
    return image_ref_dict

def clean_ref_captions(image_ref_dict):
    for key, ref_captions_lst in image_ref_dict.items():
        for item in range(len(ref_captions_lst)):
            ref_caption = ref_captions_lst[item]
            ref_caption = ref_caption.split()
            ref_caption = [word.lower() for word in ref_caption]
            ref_caption = [caption.translate(str.maketrans('', '', string.punctuation)) for caption in ref_caption]
            ref_caption = [word for word in ref_caption if word.isalpha()]
            ref_caption = [word for word in ref_caption if len(word)>1 or (word in ['a', 'i'])]
            ref_captions_lst[item] = ' '.join(ref_caption)
    # return image_ref_dict

def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    # # QUESTION 1.1
    #### In the comments below is my orginal solution for parse lines. This did not address the hanging "s" s after cleaning the captions.
    #### Additionally, a dictionary with the image ids and their cleaned captions would be useful for comparing the generated captions with the reference captions.
    #### This approach is similar to the one found here: https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
    image_ids = []
    cleaned_captions = []
    captions = []
    
    for line in lines:
        image_id = line.split(".")
        image_ids.append(image_id[0])
    
        reference = line.split("\t")
        caption = reference[1]
        captions.append(caption)

    image_ref_dict = get_image_ref_captions_dict(image_ids, captions)

    for key, ref_captions in image_ref_dict.items():
        # cleaned_captions.append(ref_captions)
        for ref_caption in ref_captions:
            cleaned_captions.append(ref_caption)
    
    return image_ids, cleaned_captions, image_ref_dict

    
    # # QUESTION 1.1
    # image_ids = []
    # cleaned_captions = []
    # bad_chars = '''!()-[]{};:`"'\,<>./?@#$%^&*_~1234567890'''
    # for line in lines:
    #     image_id = line.split(".")
    #     image_ids.append(image_id[0])
    
    #     reference = line.split("\t")
    #     caption = reference[1]
        
    #     for char in caption:
        
    #         if char in bad_chars:
    #             caption = caption.replace(char, "")
               
    #     caption = caption.strip().lower()
    #     cleaned_captions.append(caption)

    # return image_ids, cleaned_captions


def build_vocab(cleaned_captions):
    """ 
    Parses training set token file captions and builds a Vocabulary object
    Args:
        cleaned_captions (str list): cleaned list of human captions to build vocab with

    Returns:
        vocab (Vocabulary): Vocabulary object
    """

    # QUESTION 1.1
    # TODO collect words

    all_words = []
    print('Finding all words....')
    for caption in cleaned_captions:
        setence = caption.split()
        for word in setence:
            all_words.append(word)
    
    common_words = []

    print('Finding all common words...')
    
    i = 0
    for word in all_words:

        if (word not in common_words) and (all_words.count(word) > MIN_FREQUENCY):
            #print(word)
            common_words.append(word)
            i += 1
            #print("Number of common words found so far: ", i)
    print("Total number of common words found: ", len(common_words))


    # create a vocab instance
    vocab = Vocabulary()

    # add the token words
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # TODO add the rest of the words from the cleaned captions here
    
    for word in common_words:
        vocab.add_word(word)


    return vocab



def decode_caption(sampled_ids, vocab):
    """ 
    Args:
        sampled_ids (int list): list of word IDs from decoder
        vocab (Vocabulary): vocab for conversion
    Return:
        predicted_caption (str): predicted string sentence
    """
    predicted_caption = ""
    #list_sampled_ids = sampled_ids.tolist()
    # i = 0
    for idx, word_id in enumerate(sampled_ids):
        #print(word_id)
        word = vocab.idx2word[word_id]
        #print(word)
        #predicted_caption = predicted_caption + word + ' '
        # if i == 10:
        #     break
        # i = i + 1
        #predicted_caption = predicted_caption + word
        # if idx == 10:
        #     break
        # else:
        if word == '<end>':
        #if (word == '<end>') and (sampled_ids[idx-1] == word_id):
            #predicted_caption = predicted_caption[:-6]
            predicted_caption = predicted_caption[:-1]
            break
        elif (idx == 0) and (word == '<start>'):
            continue
        else:
            predicted_caption = predicted_caption + word + ' '
    # QUESTION 2.1


    return predicted_caption


"""
We need to overwrite the default PyTorch collate_fn() because our 
ground truth captions are sequential data of varying lengths. The default
collate_fn() does not support merging the captions with padding.

You can read more about it here:
https://pytorch.org/docs/stable/data.html#dataloader-collate-fn. 
"""
def caption_collate_fn(data):
    """ Creates mini-batch tensors from the list of tuples (image, caption).
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length from longest to shortest.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor).
    # if using features, 2D tensor to 3D tensor. (batch_size, 256)
    images = torch.stack(images, 0) 

    # merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]  
    #print('IMAGES', images)
    #print('TARGETS', targets)
    #print('LENGTHS', lengths)      
    return images, targets, lengths



def get_bleu_score(predicted_caption, reference_captions):
    
    predicted_words = predicted_caption.split()
    reference_words_lst = []
    for caption in reference_captions:
        reference_words = caption.split()
        reference_words_lst.append(reference_words)
    # predicted_words_lst = [predicted_words]
    #reference_words_lst = [reference_words_lst]
    #print(reference_words_lst)
    # print(predicted_words_lst)
    bleu = sentence_bleu(reference_words_lst, predicted_words, weights=(1, 0, 0, 0))
    return bleu

def display_image_captions(row_df):
    img_path = IMAGE_DIR + str(row_df.iloc[0]['image_ids']) + '.jpg'
    img = Image.open(open(img_path, 'rb'))
    show_img = plt.imshow(img)
    plt.show()
    print('Image ID: ', row_df.iloc[0]['image_ids'])
    print('Predicted caption: ', row_df.iloc[0]['predicted_caption'])
    print('Reference captions:')
    for cap in row_df.iloc[0]['reference_captions']:
        print(cap)
    print('Bleu score: ', row_df.iloc[0]['bleu_score'])
# lines = read_lines(TOKEN_FILE_TRAIN)
# # see what is in lines
# print(lines[:2])

# #########################################################################
# #
# #       QUESTION 1.1 Text preparation
# # 
# #########################################################################

# image_ids, cleaned_captions, image_ref_dict = parse_lines(lines)
# # to check the results after writing the cleaning function
# print(image_ids[:2])
# print(cleaned_captions[:2])

# vocab = build_vocab(cleaned_captions)
# # to check the results
# print("Number of words in vocab:", vocab.idx)
