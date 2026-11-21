# DIT5411-CHAN-Chun-Ki
DIT5411_Chinese_Handwriting
# DIT5411: Chinese Handwritten Character Recognition

Python and TensorFlow/Keras were used to classify **13,065 Chinese characters**. Three architectures were compared; the **Deep CNN achieved the best test accuracy of 90.01%**.

## Dataset and preprocessing

The downloaded dataset contained **684,677 images**, split into **521,860 training** and **162,817 test** images. Within each class, images were sorted by numeric filename suffix: the first 40 were used for training and the remainder for testing. For **43 classes with fewer than 41 images**, `floor(0.8 × class size)` images were used for training and the remainder for testing. This is a documented deviation from the assignment's literal first-40 rule.

Images were converted from 300 × 300 RGB to **64 × 64 grayscale** using LANCZOS resizing to reduce RAM usage and training cost within Colab resources. This may remove fine stroke details. Normalization used `1 - pixel / 255`, mapping background to 0 and strokes to 1.

## Augmentation and training

Online Keras augmentation used rotation up to **±9°**, translation up to **±5%**, and zoom factors of **0.9–1.1**, with constant background fill. Repeated original-image indices were shuffled to provide **200 augmented training views per class per epoch** (2,613,000 inputs). No separate augmented files were saved; validation and test images received no random augmentation.

Training used a T4 GPU, batch size 128, Adam (learning rate 0.0003, clipnorm 1.0), sparse categorical cross-entropy, and mixed float16 precision. Validation used 52,160 original training images, held out before augmentation. Final models were initialized afresh and refitted on all training originals using validation-selected epoch budgets before held-out testing.

## Architectures and results

All models end with **13,065 softmax outputs**. Hidden layers use LeakyReLU (slope 0.1) and classifier dropout 0.2. CNN blocks use 3 × 3 convolution followed by activation and 2 × 2 max pooling; CNN features are flattened before the Dense layer.

| Model | Architecture before output | Parameters |
|---|---|---:|
| ANN | Flatten → Dense 256 | 4,406,537 |
| Simple CNN | Conv blocks 32, 64 → Dense 128 | 3,801,481 |
| Deep CNN | Conv blocks 32, 64, 128 → Dense 128 | 2,826,761 |

| Model | Best validation | Final epochs | Test top-1 | Test top-5 | Test loss |
|---|---:|---:|---:|---:|---:|
| ANN | 34.05% | 3 | 29.52% | 52.95% | 3.8665 |
| Simple CNN | 77.00% | 3 | 74.01% | 93.15% | 1.0891 |
| Deep CNN | 92.32% | 4 | **90.01%** | **98.57%** | **0.3692** |

The Deep CNN exceeded the Simple CNN by **16.00 percentage points**. Convolutional feature extraction improved performance in this comparison, although architecture and epoch budgets both differed. Results measure image-level accuracy; writer-level separation was not verified.

## Files and references

The [notebook](DIT5411_Chinese_Handwriting.ipynb) contains experiments and outputs; the [script](DIT5411_final_training_and_test.py) implements final training/testing. `data/` contains the split, labels and exceptions; `results/` contains original full-precision metrics. Image shards and model binaries are excluded.

References: [assignment dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset); [archive used](https://github.com/chenkenanalytic/handwritting_data_all); [OpenCV reference](https://github.com/darshitajain/Introduction-to-OpenCV); [Keras project reference](https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow); [Keras documentation](https://keras.io/api/); DIT5411 ANN/CNN notes and `Image_classification_with_Deep_Learning.ipynb`.
