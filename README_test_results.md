## Final test results

Architectures and epoch budgets were fixed using validation before test evaluation. Each model was initialized afresh and refitted on all 521,860 original training images. Each epoch used 200 augmented inputs per class across 13,065 classes. The held-out test set contained 162,817 images and received no augmentation.

| Model | Epochs | Model parameters | Test top-1 accuracy | Test top-5 accuracy | Test loss |
|---|---:|---:|---:|---:|---:|
| ANN | 3 | 4,406,537 | 29.52% | 52.95% | 3.8665 |
| Simple CNN | 3 | 3,801,481 | 74.01% | 93.15% | 1.0891 |
| Deep CNN | 4 | 2,826,761 | 90.01% | 98.57% | 0.3692 |

The original split used the first 40 images in numeric filename order for training and the remainder for testing. For 43 classes with fewer than 41 images, an approximately 80/20 split was used as a documented exception. This differs from the assignment's literal first-40 rule. Validation-stage results and final test results are reported separately. Training inputs were augmented online using rotation, translation and zoom.
