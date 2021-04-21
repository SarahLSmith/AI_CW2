import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

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


def parse_lines(lines):
    """
    Parses token file captions into image_ids and captions.
    Args:
        lines (str list): str lines from token file
    Return:
        image_ids (int list): list of image ids, with duplicates
        cleaned_captions (list of lists of str): lists of words
    """
    #### will create issue for 's as these will remain as s
    image_ids = []
    cleaned_captions = []
    
    # QUESTION 1.1
    bad_chars = '''!()-[]{};:`"'\,<>./?@#$%^&*_~1234567890'''
    for line in lines:
        image_id = line.split(".")
        image_ids.append(image_id[0])
        
        reference = line.split("\t")
        caption = reference[1]
        for char in caption:
        
            if char in bad_chars:
                caption = caption.replace(char, "")
               
        caption = caption.strip().lower()
        cleaned_captions.append(caption)

    return image_ids, cleaned_captions


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
        if (word not in common_words) and (all_words.count(word) > 3):
            #print(word)
            common_words.append(word)
            i += 1
            print("Number of common words found: ", i)
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
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        #predicted_caption = predicted_caption + word
        if word == '<end>':
            break
        elif word == '<start>':
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
    return images, targets, lengths


# lines = read_lines(TOKEN_FILE_TRAIN)
# # see what is in lines
# print(lines[:2])

# #########################################################################
# #
# #       QUESTION 1.1 Text preparation
# # 
# #########################################################################

# image_ids, cleaned_captions = parse_lines(lines)
# vocab = build_vocab(cleaned_captions)
# print('Total vocab size: ', len(vocab))
# print(vocab('girl'))
# print(vocab('a'))
# print(vocab('skateboard'))
# print(len(image_ids))