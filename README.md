# Image caption generation (Flickr8k)

An image captioning model: a pretrained CNN encodes each photo, and an
RNN decoder generates a sentence describing it. Built for the COMP5623M
Artificial Intelligence coursework (2021).

## At a glance

- **Problem:** given a photo, generate a natural-language caption.
- **Data:** Flickr8k, about 8,000 images with five human captions each,
  using the provided train/test split.
- **Method:** ResNet-152 (pretrained, ImageNet) extracts 2048-d image
  features. A linear layer and batch norm map them into the embedding
  space, and a single-layer RNN decoder generates the caption word by
  word.
- **Evaluation:** average BLEU-2 and average cosine similarity between
  generated and reference captions, on the test set.
- **Result:** average BLEU-2 of **0.299** and cosine similarity of
  **0.353** on the test set. The model does well on images of dogs and
  poorly on busy scenes with people.
- **Stack:** Python, PyTorch, torchvision, scikit-learn, NLTK (BLEU).

## Skills demonstrated

- Transfer learning: reusing a pretrained CNN as a fixed feature extractor.
- Sequence modelling: a recurrent decoder with teacher forcing and
  padded, packed sequences.
- Text preprocessing: tokenising, cleaning and building a vocabulary with
  a minimum frequency cut-off.
- Evaluation: BLEU-2 for n-gram overlap, and cosine similarity on averaged
  caption vectors, with a discussion of where each one misleads.

## Results

From the coursework report:

| Metric (test set) | Score |
|---|---|
| BLEU-2 (average) | 0.2993 |
| BLEU-1 (average) | ~0.5 |
| Cosine similarity (average) | 0.3528 |

The model captions dogs well (one image scored BLEU-2 of 1.00) and
struggles with complex scenes involving people (one image scored 0.00).
The report puts this down to training data with more simple dog scenes
than complex ones. BLEU-3 and BLEU-4 were near zero, so BLEU-2 was used
as the main score.

## Method

1. **Text preparation** (`utils.py`): read the caption token files, clean
   the captions and build a vocabulary of words that appear at least
   `MIN_FREQUENCY` times. Rare words become `<unk>`.
2. **Feature extraction** (`extract_features.py`, `models.py`): pass every
   image through ResNet-152 with the final classification layer removed,
   and save the features to `features.pt`.
3. **Training** (`decoder.py`, `models.py`): the decoder takes the image
   features and the caption so far, and predicts the next word.
4. **Evaluation** (`decoder.py`): generate a caption for each test image,
   score it against its five references with BLEU-2 and cosine similarity,
   and show the highest and lowest scoring examples.

## Layout

```
cm15sls_cw2/
  config.py            paths and hyperparameters
  datasets.py          PyTorch datasets for images and captions
  extract_features.py  CNN feature extraction
  models.py            EncoderCNN (ResNet-152) and DecoderRNN
  decoder.py           training, caption generation and evaluation
  utils.py             text cleaning, vocabulary and BLEU helpers
  vocabulary.py        vocabulary class
flickr8k-custom/       captions with the train/test split (Flickr8k)
features.pt            saved image features
decoder.ckpt           trained decoder weights
```

## How to run

1. Install Python 3 with `torch`, `torchvision`, `scikit-learn`, `nltk`,
   `pandas` and `pillow`.
2. Edit `cm15sls_cw2/config.py`. `CAPTION_DIR` and `IMAGE_DIR` currently
   point at a Windows path on the original machine. Set them to your
   copy of `flickr8k-custom/captions/` and the Flickr8k image folder.
3. Extract features: `python extract_features.py`.
4. Train and evaluate: `python decoder.py`.

Known gaps in this copy:

- Six `TODO` markers are still in the code (the decoder training loop,
  the feature extraction loop, parts of `models.py` and `utils.py`).
  This copy may not reproduce the reported results. The numbers above
  come from the coursework report.
- `features.pt` and `decoder.ckpt` are saved outputs included with the
  repo. Check which run produced them before relying on them for the
  reported numbers.

## Data and credits

- **Flickr8k:** M. Hodosh, P. Young and J. Hockenmaier (2013), "Framing
  Image Description as a Ranking Task: Data, Models and Evaluation
  Metrics", *Journal of Artificial Intelligence Research* 47, 853-899.
  The image and caption files come from the course, and the caption split
  is labelled in `flickr8k-custom/`.
- **ResNet-152:** He et al. (2016), "Deep Residual Learning for Image
  Recognition", with ImageNet pretrained weights from torchvision.
- The coursework was set by the COMP5623M module team. The
  template and specification are in the module materials.
