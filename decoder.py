"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np

import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image


from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

if __name__ == '__main__':
    # if false, train model; otherwise try loading model from checkpoint and evaluate
    EVAL = True


    # reconstruct the captions and vocab, just as in extract_features.py
    lines = read_lines(TOKEN_FILE_TRAIN)
    image_ids, cleaned_captions = parse_lines(lines)
    vocab = build_vocab(cleaned_captions)


    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # initialize the models and set the learning parameters
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)


    if not EVAL:

        # load the features saved from extract_features.py
        print(len(lines))
        features = torch.load('features.pt', map_location=device)
        print("Loaded features", features.shape)

        features = features.repeat_interleave(5, 0)
        print("Duplicated features", features.shape)

        dataset_train = Flickr8k_Features(
            image_ids=image_ids,
            captions=cleaned_captions,
            vocab=vocab,
            features=features,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=64, # change as needed
            shuffle=True,
            num_workers=0, # may need to set to 0
            collate_fn=caption_collate_fn, # explicitly overwrite the collate_fn
        )


        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

        print(len(image_ids))
        print(len(cleaned_captions))
        print(features.shape)


    #########################################################################
    #
    #        QUESTION 1.3 Training DecoderRNN
    # 
    #########################################################################

        # TODO write training loop on decoder here
        for epoch in range(1, NUM_EPOCHS+1):
            
            epoch_loss = 0
            n = 0 
            for batch_idx, (inputs, captions, lengths) in enumerate(train_loader, 0):
                
                inputs = inputs.to(device)
                
                #lengths = lengths.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                targets = targets.to(device)
                captions = captions.to(device)
                optimizer.zero_grad()


                outputs = decoder.forward(inputs, captions, lengths)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n = n + 1
            print(f"Epoch {epoch} loss: {epoch_loss/n:.4f}")
        # for each batch, prepare the targets using this torch.nn.utils.rnn function
        


        # save model after training
        decoder_ckpt = torch.save(decoder, "decoder.ckpt")



    # if we already trained, and EVAL == True, reload saved model
    else:

        test_lines = read_lines(TOKEN_FILE_TEST)
        test_image_ids, test_cleaned_captions = parse_lines(test_lines)

        all_test_image_ids = test_image_ids
        test_image_ids = test_image_ids[::5]

        data_transform = transforms.Compose([ 
            transforms.Resize(224),     
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                                (0.229, 0.224, 0.225))])

        print('Preparing images....')
        dataset_test = Flickr8k_Images(
            image_ids=test_image_ids,
            transform=data_transform,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=64,
            shuffle=False,
            num_workers=2,
        )
        
        # load models
        encoder = EncoderCNN().to(device)
        decoder = torch.load("decoder.ckpt").to(device)
        encoder.eval()
        decoder.eval() # generate caption, eval mode to not influence batchnorm


    #########################################################################
    #
    #        QUESTION 2.1 Generating predictions on test data
    # 
    #########################################################################

        test_features = []

        with torch.no_grad():
            i = 0 
            for images in test_loader:
                images = images.to(device)
                output = encoder(images)
                batch_test_features = output[:,:,0,0]
                test_features.append(batch_test_features)
                i = i + 1
                print('Extracting features for batch: ', i, 'out of ', ((len(dataset_test)//64)+1))

            test_features = torch.cat(test_features)
            print('shape: ' , test_features.shape)
            #print(test_features)

        print('Generating captions...')
        sample_ids = decoder.sample(test_features)
        list_sample_ids = sample_ids.tolist()
        #print(sample_ids)

        # TODO define decode_caption() function in utils.py

        for idx, word_ids in enumerate(list_sample_ids):
            predicted_caption = decode_caption(word_ids, vocab)
            print('Predicted caption for: ', test_image_ids[idx])
            print(predicted_caption)
            print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')



    #########################################################################
    #
    #        QUESTION 2.2-3 Caption evaluation via text similarity 
    # 
    #########################################################################


        # Feel free to add helper functions to utils.py as needed,
        # documenting what they do in the code and in your report



