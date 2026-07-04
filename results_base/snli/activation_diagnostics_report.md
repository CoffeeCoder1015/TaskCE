# Activation Diagnostics Report

## RAW ACTIVATION ALPHA SWEEP

min_acts: 500

| Alpha | Zero Count | Zero % | Below Min Count | Below Min % | Kept Count | Kept % | p50 | p75 | p90 | p95 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.005 | 0 | 0.000000 | 2048 | 100.000000 | 0 | 0.000000 | 49.000000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 |
| 0.055 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 546.000000 | 549.000000 | 550.000000 | 550.000000 | 550.000000 | 550.000000 |
| 0.105 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1043.000000 | 1048.000000 | 1050.000000 | 1050.000000 | 1050.000000 | 1050.000000 |
| 0.155 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1542.000000 | 1548.000000 | 1550.000000 | 1550.000000 | 1550.000000 | 1550.000000 |

## POST-BINARIZATION ACTIVATION COUNT SUMMARY

| Metric | Value |
| :--- | :--- |
| Zero Activation Count | 0 |
| Zero Activation % | 0.000000 |
| Below Min Acts Count | 0 |
| Below Min Acts % | 0.000000 |
| Kept Count | 2048 |
| Kept % | 100.000000 |
| p50 | 546.000000 |
| p75 | 549.000000 |
| p90 | 550.000000 |
| p95 | 550.000000 |
| p99 | 550.000000 |
| Max | 550.000000 |

## SIMILARITY & CORRELATION ANALYSIS

### Pearson Correlation

#### Top Pearson Correlation Neuron Pairs

**Pearson correlation base**
```
Top positive pairs
  neuron 199 <-> neuron 1112: 0.766503
  neuron 558 <-> neuron 1096: 0.764241
  neuron 1834 <-> neuron 1970: 0.764056
  neuron 285 <-> neuron 652: 0.759452
  neuron 1404 <-> neuron 1879: 0.755893
  neuron 1404 <-> neuron 2026: 0.749719
  neuron 652 <-> neuron 1096: 0.749463
  neuron 119 <-> neuron 199: 0.745737
  neuron 1096 <-> neuron 1438: 0.743844
  neuron 758 <-> neuron 1656: 0.742695
Top negative pairs
  neuron 86 <-> neuron 201: -0.773659
  neuron 372 <-> neuron 1280: -0.764939
  neuron 1096 <-> neuron 1404: -0.762857
  neuron 264 <-> neuron 1921: -0.762383
  neuron 86 <-> neuron 1829: -0.761058
  neuron 819 <-> neuron 1054: -0.752957
  neuron 631 <-> neuron 1150: -0.752098
  neuron 372 <-> neuron 1404: -0.750292
  neuron 1150 <-> neuron 1921: -0.748444
  neuron 285 <-> neuron 1404: -0.742734
```

**Pearson correlation finetuned**
```
Top positive pairs
  neuron 657 <-> neuron 1106: 0.861603
  neuron 20 <-> neuron 1262: 0.857313
  neuron 1558 <-> neuron 1849: 0.841218
  neuron 252 <-> neuron 1146: 0.840130
  neuron 20 <-> neuron 780: 0.837775
  neuron 20 <-> neuron 1370: 0.836249
  neuron 1262 <-> neuron 1370: 0.825573
  neuron 780 <-> neuron 2032: 0.825058
  neuron 327 <-> neuron 1106: 0.824965
  neuron 444 <-> neuron 1524: 0.824896
Top negative pairs
  neuron 1445 <-> neuron 1863: -0.838959
  neuron 657 <-> neuron 1558: -0.837576
  neuron 657 <-> neuron 1801: -0.833472
  neuron 1192 <-> neuron 1849: -0.826441
  neuron 744 <-> neuron 1970: -0.824720
  neuron 1379 <-> neuron 1524: -0.824710
  neuron 327 <-> neuron 1849: -0.821421
  neuron 20 <-> neuron 1970: -0.818677
  neuron 1379 <-> neuron 1445: -0.815964
  neuron 1370 <-> neuron 1970: -0.815304
```

**Pearson correlation difference**
```
Top increased pairs
  neuron 1438 <-> neuron 1645: 1.281097
  neuron 1944 <-> neuron 1970: 1.259522
  neuron 257 <-> neuron 1006: 1.235749
  neuron 74 <-> neuron 120: 1.214160
  neuron 434 <-> neuron 1438: 1.187504
  neuron 1020 <-> neuron 1465: 1.176848
  neuron 1227 <-> neuron 1407: 1.171339
  neuron 20 <-> neuron 1407: 1.170740
  neuron 1438 <-> neuron 1859: 1.165213
  neuron 1801 <-> neuron 1944: 1.164121
Top decreased pairs
  neuron 1069 <-> neuron 1422: -1.224475
  neuron 1261 <-> neuron 1407: -1.222583
  neuron 120 <-> neuron 499: -1.202320
  neuron 252 <-> neuron 255: -1.186517
  neuron 1407 <-> neuron 1970: -1.174285
  neuron 74 <-> neuron 1394: -1.171910
  neuron 257 <-> neuron 1280: -1.163299
  neuron 423 <-> neuron 1970: -1.157421
  neuron 460 <-> neuron 741: -1.155444
  neuron 63 <-> neuron 1811: -1.152772
```

### Cosine Similarity

#### Top Cosine Similarity Neuron Pairs

**Cosine similarity base**
```
Top positive pairs
  neuron 349 <-> neuron 717: 0.996663
  neuron 278 <-> neuron 1945: 0.996205
  neuron 717 <-> neuron 1381: 0.996027
  neuron 278 <-> neuron 1049: 0.995768
  neuron 854 <-> neuron 1059: 0.995646
  neuron 278 <-> neuron 1076: 0.995578
  neuron 1435 <-> neuron 1884: 0.995564
  neuron 349 <-> neuron 901: 0.995552
  neuron 349 <-> neuron 1381: 0.995429
  neuron 237 <-> neuron 854: 0.995411
Top negative pairs
  neuron 717 <-> neuron 1049: -0.996810
  neuron 854 <-> neuron 1689: -0.996694
  neuron 1381 <-> neuron 1714: -0.996263
  neuron 1049 <-> neuron 1218: -0.996077
  neuron 278 <-> neuron 901: -0.996071
  neuron 1381 <-> neuron 1954: -0.996046
  neuron 599 <-> neuron 1884: -0.995704
  neuron 349 <-> neuron 1049: -0.995698
  neuron 485 <-> neuron 717: -0.995639
  neuron 1435 <-> neuron 1839: -0.995621
```

**Cosine similarity finetuned**
```
Top positive pairs
  neuron 1381 <-> neuron 1846: 0.988193
  neuron 195 <-> neuron 1172: 0.988123
  neuron 1218 <-> neuron 1846: 0.987791
  neuron 1028 <-> neuron 1706: 0.986989
  neuron 349 <-> neuron 764: 0.986973
  neuron 488 <-> neuron 1381: 0.986921
  neuron 349 <-> neuron 1846: 0.986591
  neuron 349 <-> neuron 1058: 0.986509
  neuron 349 <-> neuron 1218: 0.986368
  neuron 764 <-> neuron 1218: 0.986242
Top negative pairs
  neuron 139 <-> neuron 1218: -0.989387
  neuron 139 <-> neuron 1381: -0.988016
  neuron 237 <-> neuron 1218: -0.986911
  neuron 139 <-> neuron 537: -0.986694
  neuron 139 <-> neuron 764: -0.986430
  neuron 349 <-> neuron 1009: -0.986406
  neuron 525 <-> neuron 1381: -0.986383
  neuron 349 <-> neuron 1028: -0.986200
  neuron 1478 <-> neuron 1706: -0.985981
  neuron 1706 <-> neuron 1754: -0.985896
```

**Cosine similarity difference**
```
Top increased pairs
  neuron 349 <-> neuron 354: 1.949678
  neuron 354 <-> neuron 807: 1.948321
  neuron 354 <-> neuron 1807: 1.947843
  neuron 354 <-> neuron 1760: 1.944599
  neuron 354 <-> neuron 1754: 1.943523
  neuron 354 <-> neuron 1274: 1.942593
  neuron 354 <-> neuron 743: 1.940908
  neuron 354 <-> neuron 1381: 1.940236
  neuron 354 <-> neuron 1911: 1.940185
  neuron 354 <-> neuron 1846: 1.938835
Top decreased pairs
  neuron 354 <-> neuron 1884: -1.950859
  neuron 354 <-> neuron 995: -1.948200
  neuron 354 <-> neuron 613: -1.946185
  neuron 354 <-> neuron 775: -1.944376
  neuron 354 <-> neuron 1553: -1.943308
  neuron 354 <-> neuron 371: -1.941857
  neuron 237 <-> neuron 354: -1.941041
  neuron 354 <-> neuron 1009: -1.940595
  neuron 354 <-> neuron 1172: -1.940137
  neuron 1323 <-> neuron 1381: -1.939061
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

