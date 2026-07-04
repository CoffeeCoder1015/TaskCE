# Activation Diagnostics Report

## RAW ACTIVATION ALPHA SWEEP

min_acts: 500

| Alpha | Zero Count | Zero % | Below Min Count | Below Min % | Kept Count | Kept % | p50 | p75 | p90 | p95 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.002 | 0 | 0.000000 | 2048 | 100.000000 | 0 | 0.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 | 20.000000 |
| 0.052 | 0 | 0.000000 | 98 | 4.785156 | 1950 | 95.214844 | 516.000000 | 519.000000 | 520.000000 | 520.000000 | 520.000000 | 520.000000 |
| 0.102 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1014.000000 | 1018.000000 | 1020.000000 | 1020.000000 | 1020.000000 | 1020.000000 |
| 0.152 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1512.000000 | 1518.000000 | 1520.000000 | 1520.000000 | 1520.000000 | 1520.000000 |

## POST-BINARIZATION ACTIVATION COUNT SUMMARY

| Metric | Value |
| :--- | :--- |
| Zero Activation Count | 0 |
| Zero Activation % | 0.000000 |
| Below Min Acts Count | 98 |
| Below Min Acts % | 4.785156 |
| Kept Count | 1950 |
| Kept % | 95.214844 |
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
  neuron 407 <-> neuron 1608: 0.718281
  neuron 428 <-> neuron 696: 0.718251
  neuron 1608 <-> neuron 1938: 0.713179
  neuron 1350 <-> neuron 1799: 0.712603
  neuron 1359 <-> neuron 1786: 0.702947
  neuron 1595 <-> neuron 1799: 0.698852
  neuron 1350 <-> neuron 1988: 0.696029
  neuron 67 <-> neuron 599: 0.690691
  neuron 428 <-> neuron 1743: 0.689732
  neuron 668 <-> neuron 1511: 0.689537
Top negative pairs
  neuron 1791 <-> neuron 1938: -0.760702
  neuron 1799 <-> neuron 1853: -0.744553
  neuron 1104 <-> neuron 1298: -0.734643
  neuron 1350 <-> neuron 1743: -0.706383
  neuron 884 <-> neuron 1675: -0.705980
  neuron 1743 <-> neuron 1791: -0.705441
  neuron 1275 <-> neuron 1477: -0.698503
  neuron 381 <-> neuron 1162: -0.692656
  neuron 335 <-> neuron 1791: -0.691755
  neuron 86 <-> neuron 955: -0.687729
```

**Pearson correlation finetuned**
```
Top positive pairs
  neuron 1940 <-> neuron 2040: 0.764334
  neuron 980 <-> neuron 1421: 0.748564
  neuron 547 <-> neuron 590: 0.745641
  neuron 646 <-> neuron 990: 0.744096
  neuron 362 <-> neuron 1421: 0.738486
  neuron 20 <-> neuron 1316: 0.738446
  neuron 993 <-> neuron 2040: 0.734292
  neuron 1421 <-> neuron 1712: 0.731822
  neuron 1558 <-> neuron 2040: 0.731714
  neuron 126 <-> neuron 2040: 0.731304
Top negative pairs
  neuron 846 <-> neuron 2040: -0.764663
  neuron 547 <-> neuron 846: -0.763746
  neuron 1051 <-> neuron 2040: -0.751570
  neuron 1013 <-> neuron 2040: -0.750540
  neuron 1421 <-> neuron 1558: -0.748510
  neuron 646 <-> neuron 2040: -0.743513
  neuron 990 <-> neuron 2040: -0.743094
  neuron 1558 <-> neuron 1712: -0.736157
  neuron 525 <-> neuron 2040: -0.733529
  neuron 846 <-> neuron 1900: -0.727901
```

**Pearson correlation difference**
```
Top increased pairs
  neuron 208 <-> neuron 649: 1.061562
  neuron 1221 <-> neuron 1483: 1.052128
  neuron 990 <-> neuron 1438: 1.017716
  neuron 1181 <-> neuron 2040: 1.016012
  neuron 479 <-> neuron 1997: 1.014349
  neuron 547 <-> neuron 1483: 1.008827
  neuron 1172 <-> neuron 1975: 1.004218
  neuron 312 <-> neuron 951: 0.998818
  neuron 668 <-> neuron 1750: 0.997906
  neuron 1629 <-> neuron 1997: 0.997686
Top decreased pairs
  neuron 696 <-> neuron 734: -1.106951
  neuron 1263 <-> neuron 1938: -1.063631
  neuron 1221 <-> neuron 1938: -1.035651
  neuron 846 <-> neuron 1393: -1.032048
  neuron 646 <-> neuron 982: -1.022563
  neuron 261 <-> neuron 723: -1.018301
  neuron 306 <-> neuron 1648: -1.016081
  neuron 590 <-> neuron 1863: -1.012240
  neuron 275 <-> neuron 889: -1.011293
  neuron 226 <-> neuron 1172: -1.011190
```

### Cosine Similarity

#### Top Cosine Similarity Neuron Pairs

**Cosine similarity base**
```
Top positive pairs
  neuron 850 <-> neuron 1218: 0.994389
  neuron 488 <-> neuron 1381: 0.994109
  neuron 488 <-> neuron 850: 0.992838
  neuron 850 <-> neuron 1381: 0.992786
  neuron 850 <-> neuron 1803: 0.992526
  neuron 1218 <-> neuron 1381: 0.992468
  neuron 717 <-> neuron 1381: 0.992279
  neuron 901 <-> neuron 1381: 0.992268
  neuron 349 <-> neuron 1218: 0.992052
  neuron 1381 <-> neuron 1807: 0.991983
Top negative pairs
  neuron 997 <-> neuron 1381: -0.993566
  neuron 130 <-> neuron 1381: -0.993221
  neuron 130 <-> neuron 1218: -0.993130
  neuron 324 <-> neuron 1381: -0.993123
  neuron 130 <-> neuron 850: -0.992789
  neuron 1059 <-> neuron 1218: -0.992766
  neuron 1059 <-> neuron 1381: -0.992645
  neuron 613 <-> neuron 850: -0.992552
  neuron 901 <-> neuron 1059: -0.992504
  neuron 1231 <-> neuron 1381: -0.992408
```

**Cosine similarity finetuned**
```
Top positive pairs
  neuron 807 <-> neuron 1449: 0.987353
  neuron 1218 <-> neuron 1449: 0.987255
  neuron 807 <-> neuron 1218: 0.984300
  neuron 1218 <-> neuron 1760: 0.983770
  neuron 599 <-> neuron 807: 0.983249
  neuron 349 <-> neuron 1449: 0.982964
  neuron 488 <-> neuron 807: 0.982513
  neuron 807 <-> neuron 1285: 0.982497
  neuron 1449 <-> neuron 1525: 0.982334
  neuron 807 <-> neuron 1525: 0.981929
Top negative pairs
  neuron 571 <-> neuron 1218: -0.983420
  neuron 488 <-> neuron 722: -0.983102
  neuron 510 <-> neuron 1449: -0.982643
  neuron 510 <-> neuron 1218: -0.982558
  neuron 488 <-> neuron 1192: -0.982392
  neuron 510 <-> neuron 807: -0.982291
  neuron 237 <-> neuron 1218: -0.981478
  neuron 510 <-> neuron 1038: -0.980898
  neuron 510 <-> neuron 599: -0.980879
  neuron 1218 <-> neuron 1680: -0.980873
```

**Cosine similarity difference**
```
Top increased pairs
  neuron 1381 <-> neuron 1886: 1.879894
  neuron 599 <-> neuron 902: 1.877259
  neuron 237 <-> neuron 666: 1.875980
  neuron 1358 <-> neuron 1886: 1.875640
  neuron 666 <-> neuron 769: 1.874816
  neuron 666 <-> neuron 1884: 1.873939
  neuron 1760 <-> neuron 1886: 1.871929
  neuron 1218 <-> neuron 1886: 1.871298
  neuron 349 <-> neuron 1886: 1.871039
  neuron 488 <-> neuron 1520: 1.870334
Top decreased pairs
  neuron 666 <-> neuron 1218: -1.885898
  neuron 1498 <-> neuron 1886: -1.884979
  neuron 1624 <-> neuron 1886: -1.879354
  neuron 349 <-> neuron 666: -1.876667
  neuron 488 <-> neuron 1259: -1.876203
  neuron 488 <-> neuron 666: -1.874354
  neuron 1541 <-> neuron 1886: -1.872786
  neuron 510 <-> neuron 1886: -1.872421
  neuron 1680 <-> neuron 1886: -1.870801
  neuron 1231 <-> neuron 1886: -1.870273
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

