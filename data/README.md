# data/

Put the files provided with the bootcamp here (they are not committed: CIFAR-10 images and CIFAR-10N labels):

```
data/
├── train_images.npy   (50000, 32, 32, 3) uint8 – row i = index i in labels.csv
├── test_images.npy    (10000, 32, 32, 3) uint8
├── labels.csv         index, noisy_label, synthetic_label, clean_label
├── train_idx.txt      45,000 training indices
├── val_idx.txt        5,000 validation indices
└── classes.txt        10 class names
```
