# Results (Test set: 102 videos)



###### -- Oracle Mode (only classifier), Model\_TCN\_V4.pt (best model - trained for 6 epochs on 1522 videos) --

Accuracy: 0.966

F1: 0.970  

Precision: 0.952  

Recall: 0.988  

Confusion matrix:

&nbsp;\[\[ 121 8] 

&nbsp;\[  0 161]]

Balanced accuracy: 0.963



###### -- End to End pipeline, Model\_TCN\_V4.pt (best model - trained for 6 epochs on 1522 videos) -- 

Accuracy: 0.8989

F1 macro: 0.8946  |  F1 weighted: 0.8977

Precision: 0.8750  |  Recall: 0.9608

Confusion matrix:

&nbsp;\[\[ 93  21]

&nbsp;\[  6 147]]

Balanced accuracy: 0.8883

Wrong pairing (tool not in GT): 8

Missed positives: 11





##### **---- K4-Cross Validation ----**





###### -- Model\_TCN\_fold1.pt (trained for 6 epochs on 381 videos) --

Accuracy: 0.9026

F1 macro: 0.8980  |  F1 weighted: 0.9012

Precision: 0.8713  |  Recall: 0.9739

Confusion matrix:

&nbsp;\[\[ 92  22]

&nbsp;\[  4 149]]

Balanced accuracy: 0.8904

Wrong pairing (tool not in GT): 8

Missed positives: 11





###### -- Model\_TCN\_fold2.pt (trained for 6 epochs on 381 videos) --

Accuracy: 0.8689

F1 macro: 0.8604  |  F1 weighted: 0.8654

Precision: 0.8278  |  Recall: 0.9739

Confusion matrix:

 \[\[ 83  31]

 \[  4 149]]

Balanced accuracy: 0.8510

Wrong pairing (tool not in GT): 8

Missed positives: 11



###### -- Model\_TCN\_fold3.pt (trained for 6 epochs on 380 videos) --

Accuracy: 0.8614

F1 macro: 0.8530  |  F1 weighted: 0.8581

Precision: 0.8258  |  Recall: 0.9608

Confusion matrix:

&nbsp;\[\[ 83  31]

&nbsp;\[  6 147]]

Balanced accuracy: 0.8444

Wrong pairing (tool not in GT): 8

Missed positives: 12



###### -- Model\_TCN\_fold4.pt (trained for 6 epochs on 380 videos)--

Accuracy: 0.8764

F1 macro: 0.8698  |  F1 weighted: 0.8741

Precision: 0.8448  |  Recall: 0.9608

Confusion matrix:

&nbsp;\[\[ 87  27]

&nbsp;\[  6 147]]

Balanced accuracy: 0.8620

Wrong pairing (tool not in GT): 8

Missed positives: 12





##### --- Final Results ---

Accuracy: 0.8773 +/- 0.0159

F1 macro: 0.8703 +/- 0.0178

F1 weighted: 0.8747 +/- 0.0160

Precision: 0.8424 +/- 0.0202

Recall: 0.9674 +/- 0.0072

Balanced accuracy: 0.8619 +/- 0.0179






### Benchmark High Quality vs Near Real Time

| Profile | Parameters | YOLO (ms / fps) | Depth (ms / fps) | Preproc (ms / fps) | TCN (ms / fps) | Totale (ms / fps) |
|---------|-----------|-----------------|------------------|--------------------|----------------|-------------------|
| **HQ** | YOLO imgsz=640<br>Depth full-res, Depth every frame,<br>SEQ_LEN=5, IMG_SIZE=224 | 20–25 ms<br>(40–50 fps) | 40–90 ms<br>(10–20 fps) | 30–60 ms<br>(16–33 fps) | 8–15 ms<br>(65–120 fps) | 110–160 ms<br>(6–9 fps) |
| **NRT** | YOLO imgsz=480<br>Depth downscale=0.5,<br>Depth every 3 frame,<br>SEQ_LEN=5, IMG_SIZE=224 | 12–15 ms<br>(65–80 fps) | 15–25 ms<br>(40–65 fps)<br> | 10–15 ms<br>(65–100 fps) | 6–10 ms<br>(100–160 fps) | 35–45 ms<br>(22–28 fps) |

### Note
- **Total** = mean value per frame for the entire pipeline.  
- For **surgical real-time** we need ≤33–40 ms/frame (≥25–30 fps).  
- HQ more accurate, but slower (≈8 fps).  
- NRT reduces the depth quality and the calculus frequenzy, but it is quite real-time (≈25 fps) and it is stil suffciently accurate.  
