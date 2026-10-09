# Report – Learning from Messy Human Labels (CIFAR-10N worst)

Q1. Which class pairs account for most of the noise?

The training labels are 40.24% wrong (18,109 of 45,000), and the errors are concentrated, not uniform. Counting both directions, automobile↔truck (13.1% of all errors) and cat↔dog (9.7%) dominate, followed by deer↔horse (5.1%) and airplane↔ship (4.1%). Together these four pairs cover 32% of the errors. A quarter of automobiles are labelled "truck", 28% of trucks "automobile", and 21% of cats "dog". Cat and deer are the noisiest classes (51% wrong), horse the cleanest (28%). The synthetic labels, by contrast, spread errors evenly (about 4–5% per wrong class).

Q2. At which epoch does memorization of wrong labels start, and what is the evidence?

Memorization starts at epoch 38. Until about epoch 37, accuracy on the mislabeled subset (agreement with the wrong label) stays flat at 12–15%; the net still predicts the true class for ~65% of them, learning the clean pattern. From epoch 38, as the learning rate decays, agreement with wrong labels climbs to 64% by epoch 60 and clean validation accuracy drops from 79.0% (epoch 45) to 70.5%. Noisy validation accuracy peaks at epoch 43 (55.9%), our early-stop epoch.

Q3. Which noise-score signal worked best? Describe one failure mode.

AUM worked best (AUROC 0.966, precision@k 0.882): the largest other-class probability minus the noisy label's probability, averaged over epochs 1–43 of the noisy run, before memorization. Cross-validated held-out loss reached 0.938. Feature-space kNN also works (an image whose 50 nearest neighbours mostly carry another label is suspect): AUROC 0.93–0.97, but only on features from a non-memorizing network (memorized: 0.75; hand-crafted LBP/GLCM/HOG: 0.83). Failure mode: look-alike errors; recall is 79% on automobile↔truck and cat↔dog versus 91% elsewhere.

Q4. Why did your best Task 4 method help, or not help?

ELR + mixup (2 nets) won on noisy validation: 59.2% (89.5% clean) versus 55.9% (77.1%) for early-stopped CE. ELR keeps a running average of each image's past predictions and adds log(1 − ⟨p, target⟩) to the loss. Early predictions mostly reflect the true class, so this cancels the gradient pulling the net toward wrong labels. Mixup blends image pairs, so single labels are hard to memorize. Fit to wrong labels falls to 12% (CE: 64%), and 79% of mislabeled images get their true class. Limit: truck↔automobile confusions remain.

Q5. Did the ranking of methods change between real and synthetic noise? Why?

No: ELR+mixup > relabel+mixup > GCE+mixup > CE on both (synthetic val 54.6 / 54.1 / 53.7 / 50.2%). Synthetic noise is easier (clean val 90–92.5% versus 85–89.5%; early-stopped CE 83% versus 77%): random wrong labels contradict the image. The methods matter more on human noise (best minus CE: +12.4 versus +9.2 clean points), because human errors are plausible look-alikes on hard images: humans mislabel 52% of the hardest fifth of validation images (clean-model confidence) versus 37% of the rest, while random noise is flat (44% versus 40%).
