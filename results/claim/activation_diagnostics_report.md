# Activation Diagnostics Report

## RAW ACTIVATION ALPHA SWEEP

min_acts: 500

| Alpha | Zero Count | Zero % | Below Min Count | Below Min % | Kept Count | Kept % | p50 | p75 | p90 | p95 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.002 | 0 | 0.000000 | 2048 | 100.000000 | 0 | 0.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 |
| 0.052 | 0 | 0.000000 | 93 | 4.541016 | 1955 | 95.458984 | 516.000000 | 519.000000 | 520.000000 | 520.000000 | 520.000000 | 520.000000 |
| 0.102 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1014.000000 | 1018.000000 | 1020.000000 | 1020.000000 | 1020.000000 | 1020.000000 |
| 0.152 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1513.000000 | 1517.000000 | 1520.000000 | 1520.000000 | 1520.000000 | 1520.000000 |

## POST-BINARIZATION ACTIVATION COUNT SUMMARY

| Metric | Value |
| :--- | :--- |
| Zero Activation Count | 0 |
| Zero Activation % | 0.000000 |
| Below Min Acts Count | 93 |
| Below Min Acts % | 4.541016 |
| Kept Count | 1955 |
| Kept % | 95.458984 |
| p50 | 516.000000 |
| p75 | 519.000000 |
| p90 | 520.000000 |
| p95 | 520.000000 |
| p99 | 520.000000 |
| Max | 520.000000 |

## SIMILARITY & CORRELATION ANALYSIS

### Pearson Correlation

#### Top Pearson Correlation Neuron Pairs

**Pearson correlation base**
```
Top positive pairs
  neuron 428 <-> neuron 696: 0.718078
  neuron 407 <-> neuron 1608: 0.716760
  neuron 1608 <-> neuron 1938: 0.712780
  neuron 1350 <-> neuron 1799: 0.711818
  neuron 1359 <-> neuron 1786: 0.701264
  neuron 1595 <-> neuron 1799: 0.698764
  neuron 1350 <-> neuron 1988: 0.695917
  neuron 668 <-> neuron 1511: 0.690229
  neuron 67 <-> neuron 599: 0.689571
  neuron 428 <-> neuron 1743: 0.688620
Top negative pairs
  neuron 1791 <-> neuron 1938: -0.759246
  neuron 1799 <-> neuron 1853: -0.744895
  neuron 1104 <-> neuron 1298: -0.734412
  neuron 884 <-> neuron 1675: -0.706924
  neuron 1350 <-> neuron 1743: -0.705692
  neuron 1743 <-> neuron 1791: -0.704853
  neuron 1275 <-> neuron 1477: -0.697366
  neuron 381 <-> neuron 1162: -0.693622
  neuron 335 <-> neuron 1791: -0.691508
  neuron 86 <-> neuron 955: -0.686510
```

**Pearson correlation finetuned**
```
Top positive pairs
  neuron 1940 <-> neuron 2040: 0.764293
  neuron 980 <-> neuron 1421: 0.749168
  neuron 547 <-> neuron 590: 0.745555
  neuron 646 <-> neuron 990: 0.744435
  neuron 20 <-> neuron 1316: 0.738842
  neuron 362 <-> neuron 1421: 0.738617
  neuron 993 <-> neuron 2040: 0.734150
  neuron 1421 <-> neuron 1712: 0.731648
  neuron 126 <-> neuron 2040: 0.731392
  neuron 1558 <-> neuron 2040: 0.731215
Top negative pairs
  neuron 846 <-> neuron 2040: -0.764885
  neuron 547 <-> neuron 846: -0.763325
  neuron 1051 <-> neuron 2040: -0.751468
  neuron 1013 <-> neuron 2040: -0.750846
  neuron 1421 <-> neuron 1558: -0.748499
  neuron 646 <-> neuron 2040: -0.743733
  neuron 990 <-> neuron 2040: -0.743487
  neuron 1558 <-> neuron 1712: -0.735971
  neuron 525 <-> neuron 2040: -0.733848
  neuron 846 <-> neuron 1900: -0.728193
```

**Pearson correlation difference**
```
Top increased pairs
  neuron 208 <-> neuron 649: 1.060400
  neuron 1221 <-> neuron 1483: 1.053052
  neuron 990 <-> neuron 1438: 1.018007
  neuron 1181 <-> neuron 2040: 1.015525
  neuron 479 <-> neuron 1997: 1.015040
  neuron 547 <-> neuron 1483: 1.006600
  neuron 1172 <-> neuron 1975: 1.004201
  neuron 995 <-> neuron 1921: 0.999631
  neuron 312 <-> neuron 951: 0.999366
  neuron 668 <-> neuron 1750: 0.997745
Top decreased pairs
  neuron 696 <-> neuron 734: -1.106258
  neuron 1263 <-> neuron 1938: -1.062130
  neuron 1221 <-> neuron 1938: -1.035349
  neuron 846 <-> neuron 1393: -1.031044
  neuron 646 <-> neuron 982: -1.021870
  neuron 261 <-> neuron 723: -1.019075
  neuron 306 <-> neuron 1648: -1.018970
  neuron 275 <-> neuron 889: -1.011415
  neuron 590 <-> neuron 1863: -1.010746
  neuron 226 <-> neuron 1172: -1.010420
```

### Cosine Similarity

#### Top Cosine Similarity Neuron Pairs

**Cosine similarity base**
```
Top positive pairs
  neuron 850 <-> neuron 1218: 0.994385
  neuron 488 <-> neuron 1381: 0.994100
  neuron 488 <-> neuron 850: 0.992844
  neuron 850 <-> neuron 1381: 0.992810
  neuron 850 <-> neuron 1803: 0.992538
  neuron 1218 <-> neuron 1381: 0.992470
  neuron 717 <-> neuron 1381: 0.992264
  neuron 901 <-> neuron 1381: 0.992259
  neuron 349 <-> neuron 1218: 0.992067
  neuron 1381 <-> neuron 1807: 0.991991
Top negative pairs
  neuron 997 <-> neuron 1381: -0.993582
  neuron 130 <-> neuron 1381: -0.993250
  neuron 324 <-> neuron 1381: -0.993118
  neuron 130 <-> neuron 1218: -0.993110
  neuron 130 <-> neuron 850: -0.992791
  neuron 1059 <-> neuron 1218: -0.992742
  neuron 1059 <-> neuron 1381: -0.992640
  neuron 613 <-> neuron 850: -0.992546
  neuron 901 <-> neuron 1059: -0.992544
  neuron 1231 <-> neuron 1381: -0.992399
```

**Cosine similarity finetuned**
```
Top positive pairs
  neuron 807 <-> neuron 1449: 0.987392
  neuron 1218 <-> neuron 1449: 0.987252
  neuron 807 <-> neuron 1218: 0.984318
  neuron 1218 <-> neuron 1760: 0.983761
  neuron 599 <-> neuron 807: 0.983231
  neuron 349 <-> neuron 1449: 0.982959
  neuron 807 <-> neuron 1285: 0.982506
  neuron 488 <-> neuron 807: 0.982485
  neuron 1449 <-> neuron 1525: 0.982319
  neuron 807 <-> neuron 1525: 0.981902
Top negative pairs
  neuron 571 <-> neuron 1218: -0.983429
  neuron 488 <-> neuron 722: -0.983141
  neuron 510 <-> neuron 1449: -0.982651
  neuron 510 <-> neuron 1218: -0.982579
  neuron 488 <-> neuron 1192: -0.982389
  neuron 510 <-> neuron 807: -0.982300
  neuron 237 <-> neuron 1218: -0.981436
  neuron 510 <-> neuron 1038: -0.980906
  neuron 1218 <-> neuron 1680: -0.980872
  neuron 510 <-> neuron 599: -0.980839
```

**Cosine similarity difference**
```
Top increased pairs
  neuron 1381 <-> neuron 1886: 1.880024
  neuron 599 <-> neuron 902: 1.877326
  neuron 237 <-> neuron 666: 1.876477
  neuron 1358 <-> neuron 1886: 1.875701
  neuron 666 <-> neuron 769: 1.875334
  neuron 666 <-> neuron 1884: 1.874246
  neuron 1760 <-> neuron 1886: 1.871907
  neuron 1218 <-> neuron 1886: 1.871371
  neuron 349 <-> neuron 1886: 1.871009
  neuron 488 <-> neuron 1520: 1.870219
Top decreased pairs
  neuron 666 <-> neuron 1218: -1.886452
  neuron 1498 <-> neuron 1886: -1.885083
  neuron 1624 <-> neuron 1886: -1.879627
  neuron 349 <-> neuron 666: -1.877202
  neuron 488 <-> neuron 1259: -1.876136
  neuron 488 <-> neuron 666: -1.874809
  neuron 1541 <-> neuron 1886: -1.872840
  neuron 510 <-> neuron 1886: -1.872396
  neuron 1680 <-> neuron 1886: -1.870878
  neuron 1231 <-> neuron 1886: -1.870480
```

## VISUALIZATIONS

### Post-Binarization Activation Count Histograms

| Full Histogram | Nonzero Histogram |
| :---: | :---: |
| ![Post-Binarization Activation Counts](./activation_counts_hist_full.png) | ![Post-Binarization Nonzero Activation Counts](./activation_counts_hist_nonzero.png) |

### Binarized Activation Jaccard Similarity

#### Jaccard Similarity / IoU Heatmap

![Jaccard Similarity / IoU Heatmap](./binarized_activation_jaccard_heatmap.png)

### Pearson Correlation Heatmaps

| Base Heatmap | Finetuned Heatmap | Difference Heatmap |
| :---: | :---: | :---: |
| ![Pearson Correlation Base](./raw_activation_correlation_heatmap_base.png) | ![Pearson Correlation Finetuned](./raw_activation_correlation_heatmap_finetuned.png) | ![Pearson Correlation Difference](./raw_activation_correlation_heatmap_difference.png) |

### Cosine Similarity Heatmaps

| Base Heatmap | Finetuned Heatmap | Difference Heatmap |
| :---: | :---: | :---: |
| ![Cosine Similarity Base](./raw_activation_cosine_similarity_heatmap_base.png) | ![Cosine Similarity Finetuned](./raw_activation_cosine_similarity_heatmap_finetuned.png) | ![Cosine Similarity Difference](./raw_activation_cosine_similarity_heatmap_difference.png) |

