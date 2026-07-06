# Activation Diagnostics Report

## RAW ACTIVATION ALPHA SWEEP

min_acts: 500

| Alpha | Zero Count | Zero % | Below Min Count | Below Min % | Kept Count | Kept % | p50 | p75 | p90 | p95 | p99 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.005 | 0 | 0.000000 | 2048 | 100.000000 | 0 | 0.000000 | 49.000000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 | 50.000000 |
| 0.055 | 0 | 0.000000 | 15 | 0.732422 | 2033 | 99.267578 | 544.000000 | 548.000000 | 550.000000 | 550.000000 | 550.000000 | 550.000000 |
| 0.105 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1041.000000 | 1047.000000 | 1049.000000 | 1050.000000 | 1050.000000 | 1050.000000 |
| 0.155 | 0 | 0.000000 | 0 | 0.000000 | 2048 | 100.000000 | 1539.000000 | 1547.000000 | 1549.000000 | 1550.000000 | 1550.000000 | 1550.000000 |

## POST-BINARIZATION ACTIVATION COUNT SUMMARY

| Metric | Value |
| :--- | :--- |
| Zero Activation Count | 0 |
| Zero Activation % | 0.000000 |
| Below Min Acts Count | 15 |
| Below Min Acts % | 0.732422 |
| Kept Count | 2033 |
| Kept % | 99.267578 |
| p50 | 544.000000 |
| p75 | 548.000000 |
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
  neuron 710 <-> neuron 1791: 0.832753
  neuron 237 <-> neuron 1686: 0.830797
  neuron 962 <-> neuron 1818: 0.828845
  neuron 144 <-> neuron 789: 0.828296
  neuron 718 <-> neuron 1972: 0.822931
  neuron 683 <-> neuron 1686: 0.818347
  neuron 311 <-> neuron 433: 0.815609
  neuron 493 <-> neuron 898: 0.815165
  neuron 10 <-> neuron 1964: 0.812006
  neuron 240 <-> neuron 710: 0.807793
Top negative pairs
  neuron 370 <-> neuron 1686: -0.860433
  neuron 1519 <-> neuron 1964: -0.860270
  neuron 747 <-> neuron 1051: -0.824828
  neuron 28 <-> neuron 789: -0.814943
  neuron 1461 <-> neuron 2035: -0.812240
  neuron 237 <-> neuron 1235: -0.809610
  neuron 123 <-> neuron 1778: -0.809604
  neuron 240 <-> neuron 888: -0.809418
  neuron 898 <-> neuron 1719: -0.807669
  neuron 493 <-> neuron 1719: -0.805525
```

**Pearson correlation finetuned**
```
Top positive pairs
  neuron 915 <-> neuron 1888: 0.967067
  neuron 696 <-> neuron 748: 0.963160
  neuron 696 <-> neuron 1633: 0.960481
  neuron 781 <-> neuron 1527: 0.959882
  neuron 781 <-> neuron 1335: 0.959087
  neuron 696 <-> neuron 1888: 0.957991
  neuron 1527 <-> neuron 1943: 0.956981
  neuron 1061 <-> neuron 1335: 0.956648
  neuron 1527 <-> neuron 1911: 0.956284
  neuron 915 <-> neuron 1475: 0.955171
Top negative pairs
  neuron 1527 <-> neuron 1888: -0.969664
  neuron 257 <-> neuron 1527: -0.966670
  neuron 696 <-> neuron 1527: -0.966213
  neuron 696 <-> neuron 781: -0.962611
  neuron 781 <-> neuron 1888: -0.961248
  neuron 781 <-> neuron 915: -0.961166
  neuron 915 <-> neuron 1205: -0.960230
  neuron 97 <-> neuron 1633: -0.958950
  neuron 1527 <-> neuron 1633: -0.955050
  neuron 748 <-> neuron 1527: -0.954786
```

**Pearson correlation difference**
```
Top increased pairs
  neuron 789 <-> neuron 1136: 1.600636
  neuron 269 <-> neuron 710: 1.578811
  neuron 751 <-> neuron 986: 1.550897
  neuron 986 <-> neuron 1924: 1.547029
  neuron 962 <-> neuron 1786: 1.545558
  neuron 50 <-> neuron 1924: 1.539343
  neuron 1532 <-> neuron 1834: 1.537985
  neuron 1127 <-> neuron 1987: 1.534759
  neuron 701 <-> neuron 1008: 1.526431
  neuron 440 <-> neuron 1633: 1.526141
Top decreased pairs
  neuron 135 <-> neuron 751: -1.588360
  neuron 433 <-> neuron 701: -1.587076
  neuron 2 <-> neuron 710: -1.579258
  neuron 2 <-> neuron 86: -1.571701
  neuron 177 <-> neuron 536: -1.568336
  neuron 440 <-> neuron 778: -1.566951
  neuron 269 <-> neuron 1662: -1.565816
  neuron 269 <-> neuron 1834: -1.562498
  neuron 751 <-> neuron 1435: -1.561480
  neuron 2 <-> neuron 240: -1.545434
```

### Cosine Similarity

#### Top Cosine Similarity Neuron Pairs

**Cosine similarity base**
```
Top positive pairs
  neuron 318 <-> neuron 1455: 0.997023
  neuron 318 <-> neuron 1146: 0.997020
  neuron 318 <-> neuron 1492: 0.996480
  neuron 318 <-> neuron 329: 0.996358
  neuron 394 <-> neuron 434: 0.996352
  neuron 318 <-> neuron 1386: 0.995915
  neuron 1386 <-> neuron 1492: 0.995905
  neuron 318 <-> neuron 1959: 0.995887
  neuron 1386 <-> neuron 1455: 0.995673
  neuron 1455 <-> neuron 1492: 0.995659
Top negative pairs
  neuron 394 <-> neuron 573: -0.996842
  neuron 318 <-> neuron 1923: -0.996328
  neuron 425 <-> neuron 912: -0.996231
  neuron 1492 <-> neuron 1923: -0.995908
  neuron 454 <-> neuron 1709: -0.995880
  neuron 169 <-> neuron 318: -0.995783
  neuron 318 <-> neuron 425: -0.995743
  neuron 978 <-> neuron 1923: -0.995614
  neuron 978 <-> neuron 1865: -0.995505
  neuron 958 <-> neuron 1107: -0.995494
```

**Cosine similarity finetuned**
```
Top positive pairs
  neuron 634 <-> neuron 1038: 0.995988
  neuron 985 <-> neuron 987: 0.995275
  neuron 1038 <-> neuron 1602: 0.994872
  neuron 1441 <-> neuron 1919: 0.994792
  neuron 634 <-> neuron 698: 0.994777
  neuron 423 <-> neuron 1038: 0.994636
  neuron 434 <-> neuron 2001: 0.994594
  neuron 634 <-> neuron 1871: 0.994487
  neuron 318 <-> neuron 526: 0.994465
  neuron 698 <-> neuron 1925: 0.994442
Top negative pairs
  neuron 614 <-> neuron 698: -0.995422
  neuron 558 <-> neuron 698: -0.995355
  neuron 614 <-> neuron 1871: -0.994970
  neuron 1843 <-> neuron 1871: -0.994949
  neuron 698 <-> neuron 985: -0.994863
  neuron 1197 <-> neuron 1428: -0.994832
  neuron 558 <-> neuron 634: -0.994790
  neuron 614 <-> neuron 634: -0.994788
  neuron 103 <-> neuron 698: -0.994656
  neuron 423 <-> neuron 1090: -0.994616
```

**Cosine similarity difference**
```
Top increased pairs
  neuron 1331 <-> neuron 1919: 1.974248
  neuron 153 <-> neuron 1867: 1.970403
  neuron 697 <-> neuron 1307: 1.969614
  neuron 1331 <-> neuron 1889: 1.967321
  neuron 153 <-> neuron 549: 1.967046
  neuron 140 <-> neuron 821: 1.966988
  neuron 558 <-> neuron 1307: 1.966392
  neuron 1307 <-> neuron 2047: 1.965242
  neuron 1734 <-> neuron 1918: 1.964870
  neuron 446 <-> neuron 1307: 1.964422
Top decreased pairs
  neuron 1267 <-> neuron 1331: -1.974114
  neuron 821 <-> neuron 1957: -1.972138
  neuron 1709 <-> neuron 1918: -1.970796
  neuron 153 <-> neuron 698: -1.969836
  neuron 153 <-> neuron 1159: -1.968524
  neuron 804 <-> neuron 821: -1.968373
  neuron 318 <-> neuron 1307: -1.968218
  neuron 821 <-> neuron 1290: -1.967979
  neuron 1331 <-> neuron 1938: -1.967822
  neuron 153 <-> neuron 1987: -1.967690
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

