"""
COMP5623M Coursework on Image Caption Generation


python decoder.py


"""

import torch
import numpy as np
import pandas as pd

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
    image_ids, cleaned_captions, image_ref_dict = parse_lines(lines)
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
        test_image_ids, test_cleaned_captions, test_image_ref_dict = parse_lines(test_lines)

        all_test_image_ids = test_image_ids
        single_test_image_ids = test_image_ids[::5]

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
            num_workers=0,
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
                output = encoder.forward(images)
                batch_test_features = output[:,:,0,0]
                test_features.append(batch_test_features)
                i = i + 1
                print('Extracting features for batch: ', i, 'out of ', ((len(dataset_test)//64)+1))
            #print(test_features)
            all_test_features = torch.cat(test_features)
            print('shape: ' , all_test_features.shape)
            #print(test_features)
        #all_test_features = all_test_features.repeat_interleave(5, 0)
        dataset_test = Flickr8k_Features(
            image_ids=all_test_image_ids,
            captions=test_cleaned_captions,
            vocab=vocab,
            features=all_test_features,
        )

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=caption_collate_fn,
        )

    #     test_img_feats = dataset_test.features
    #     print(test_img_feats.shape)
    #     #test_img_feats = test_img_feats.to(device)
    #     #sample_ids = decoder.sample(test_img_feats)
    #     sample_ids = decoder.sample(all_test_features)
    #     list_sample_ids = sample_ids.tolist()
    # # print(test_loader.dataset_test[0])
    # # for batch_idx, feats in enumerate(dataset_test.features):
    # #     sample_ids = decoder.sample(feats)
    # #     list_sample_ids = sample_ids.tolist()
    #     targets = dataset_test.captions.tolist()

    #     for idx, word_id in enumerate(list_sample_ids):

    #         predicted_caption = decode_caption(word_id, vocab)
    #         print(predicted_caption)
    #         #print(captions[idx])
            
    #         target = targets[idx]
    #         caption = ''
    #         for i, cap_word_id in enumerate(target):
    #             word = vocab.idx2word[cap_word_id]
    #             if i < len(target):
    #                 caption = caption + word + ' '
    #             else:
    #                 caption = caption + word
    #         print(caption)
        #print(images)
        #print(captions)
            #break
        # break

        # index = 0
        # tars = []
        # for i in range(len(dataset_test)):
        #     feat, tar = dataset_test[i]
        #     if (i == 0) or (i%5 == 0):
        #         sample_ids = decoder.sample(feat)
        #         list_sample_ids = sample_ids.tolist()
        #         predicted_caption = decode_caption(list_sample_ids, vocab)
        #         print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        #         print('Predicted caption for image: ', dataset_test.image_ids[i])
        #         print('')
        #         print(predicted_caption)
        #         print('')
        #         print('Reference captions: ')
        #     else:
        #         target = tar.tolist()
        #         caption = ''
        #         for idx, word_id in enumerate(target):
        #             word = vocab.idx2word[word_id]
        #             if idx < len(target):
        #                 caption = caption + word + ' '
        #             else:
        #                 caption = caption + word
        #         print(caption)

            #print('Features: ', feat)
            #print('Target :', tar)
        #print(dataset_test[0])
        # for idx, (inputs, captions, lengths) in enumerate(test_loader, 0):
        #     image_feats = inputs.to(device)

        print('Generating captions...')
        index = 0
        captions = []
        bleu_scores = []
        low_bleu = 0.01
        high_bleu = 0.5
        img_ref_dict = dict()
        img_cap_pred_dict = dict()
        img_bleu_dict = dict()
        for batch_idx, (feats, caps, length) in enumerate(test_loader):
            test_feats = feats.to(device)
            sample_ids = decoder.sample(test_feats)
            list_sample_ids = sample_ids.tolist()
            targets = caps.tolist()
            if (index%10 == 4) or (index%10 == 9):

                captions.append(targets[0])
                predicted_caption = decode_caption(list_sample_ids[0], vocab)
                print('Predicted caption for image: ', test_image_ids[index])
                print('')
                print(predicted_caption)
                print('')
                print('Reference captions: ')
                ref_captions = []
                for i,  caption in enumerate(captions):
                    ref_cap = decode_caption(caption, vocab)
                    ref_captions.append(ref_cap)
                    print(ref_cap)
                
                bleu = get_bleu_score(predicted_caption, ref_captions)
                print('')
                print('Bleu score: ', bleu)
                bleu_scores.append(bleu)
                print('')
                image_ref_dict[test_image_ids[index]] = ref_captions
                img_cap_pred_dict[test_image_ids[index]] = predicted_caption
                img_bleu_dict[test_image_ids[index]] = bleu
                
                #reference_cap = decode_caption(targets[0], vocab)
                #print(reference_cap)
                captions = []
                
                print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            else:
                captions.append(targets[0])
            index = index + 1
        print('Average BLEU score: ', (sum(bleu_scores)/len(bleu_scores)))
        df = pd.DataFrame({'image_ids': single_test_image_ids})
        df['predicted_caption'] = df['image_ids'].map(img_cap_pred_dict)
        df['reference_captions'] = df['image_ids'].map(image_ref_dict)
        df['bleu_score'] = df['image_ids'].map(img_bleu_dict)
        print(df)
        #display_image_captions(df.iloc[[0]])
        low_df = df.sort_values(by=['bleu_score'], ascending=False)
        display_image_captions(low_df.iloc[[0]])
        high_df = df.sort_values(by=['bleu_score'], ascending=True)
        display_image_captions(high_df.iloc[[0]])
        # all_test_features = all_test_features.to(device)
        # sample_ids = decoder.sample(all_test_features)
        # print('Converting to list')
        # list_sample_ids = sample_ids.tolist()
        #print(sample_ids)

        # TODO define decode_caption() function in utils.py
            
            # for idx, word_ids in enumerate(list_sample_ids):
            #     predicted_caption = decode_caption(word_ids, vocab)
            #     #print('Predicted caption for image: ', test_image_ids[index])
            #     print('')
            #     print(predicted_caption)
            #     print('')
            #     print('Reference captions: ')
            #     reference_cap = decode_caption(targets[idx], vocab)
            #     print(reference_cap)
            #     #print(test_cleaned_captions[index])
            #     # print(test_cleaned_captions[index+1])
            #     # print(test_cleaned_captions[index+2])
            #     # print(test_cleaned_captions[index+3])
            #     # print(test_cleaned_captions[index+4])
            #     #references = [test_cleaned_captions[index], test_cleaned_captions[index+1], test_cleaned_captions[index+2], test_cleaned_captions[index+3], test_cleaned_captions[index+4]]
            #     #bleu = get_bleu_score(predicted_caption, references)
            #     #print('Bleu score: ', bleu)
            #     #index = index + 5
            #     index = index + 1
            #     print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            #     #break


    #########################################################################
    #
    #        QUESTION 2.2-3 Caption evaluation via text similarity 
    # 
    #########################################################################


        # Feel free to add helper functions to utils.py as needed,
        # documenting what they do in the code and in your report




