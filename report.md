# Report – Learning from Messy Human Labels (CIFAR-10N worst)

Q1. Which class pairs account for most of the noise?

The training labels are 40.24% wrong (18,109 of 45,000), and the errors are concentrated, not uniform. Counting both directions, automobile↔truck (13.1% of all errors) and cat↔dog (9.7%) dominate, followed by deer↔horse (5.1%) and airplane↔ship (4.1%). Together these four pairs cover 32% of the errors. A quarter of automobiles are labelled "truck", 28% of trucks "automobile", and 21% of cats "dog". Cat and deer are the noisiest classes (51% wrong), horse the cleanest (28%). The synthetic labels, by contrast, spread errors evenly (about 4–5% per wrong class).

Q2. At which epoch does memorization of wrong labels start, and what is the evidence?

Memorization starts at epoch 38. Until about epoch 37, accuracy on the mislabeled subset (agreement with the wrong label) stays flat at 12–15%; the net still predicts the true class for ~65% of them, learning the clean pattern. From epoch 38, as the learning rate decays, agreement with wrong labels climbs to 64% by epoch 60 and clean validation accuracy drops from 79.0% (epoch 45) to 70.5%. Noisy validation accuracy peaks at epoch 43 (55.9%), our early-stop epoch.

Q3. Which noise-score signal worked best? Describe one failure mode.

AUM worked best (AUROC 0.966, precision@k 0.882): the largest other-class probability minus the noisy label's probability, averaged over epochs 1–43 of the noisy run, before memorization. Cross-validated held-out loss reached 0.938. Feature-space kNN also works (an image whose 50 nearest neighbours mostly carry another label is suspect): AUROC 0.93–0.97, but only on features from a non-memorizing network (memorized: 0.75; hand-crafted LBP/GLCM/HOG: 0.83). Failure mode: look-alike errors; recall is 79% on automobile↔truck and cat↔dog versus 91% elsewhere.

Q4. Why did your best Task 4 method help, or not help?

Best on noisy validation: ELR + mixup (2 nets) with class rebalancing, 59.3% (89.9% clean) versus 55.9% (77.1%) for early-stopped CE. ELR adds log(1 − ⟨p, target⟩), the target being a running average of the image's past, mostly correct predictions, so wrong labels lose their pull; mixup stops single labels being memorized (wrong-label fit 64% → 12%). Annotators over-use "dog": the nets predicted dog for 13% and cat for 7% of training images. Shifting outputs toward balanced classes lifts cat accuracy from 64% to 68%.

Q5. Did the ranking of methods change between real and synthetic noise? Why?

Barely: ELR+rebalancing leads on both (synthetic val 54.7% vs CE 50.2%); relabelling and distribution-aligned ELR swap 2nd/3rd within noise. Synthetic noise is easier (clean val 91.5–92.6% versus 87.7–90.6%; early-stopped CE 83% versus 77%): random wrong labels contradict the image. The methods matter more on human noise (+12.8 versus +9.2 clean points over CE), because human errors are plausible look-alikes on hard images: humans mislabel 52% of the hardest fifth of validation images versus 37% of the rest; random noise is flat (44% versus 40%).
