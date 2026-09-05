# Duplicate / leakage report: FGSCR

threshold: Hamming <= 5 on **both** aHash and pHash, over variants ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip'], at two scales (full frame and centre 60% crop).

A full-frame match with a **non**-matching centre crop means the two images share a background but hold different targets -- a compositing artefact, not a duplicate. Those are counted separately below and are **not** merged into duplicate groups.

- images scanned: **7778**
- duplicate pairs found: **4838**
- images belonging to a duplicate group: **3703** (47.6%)
- unique images after grouping: **5233**
- largest duplicate group: **9** images
- groups with >1 member: 1158
- **cross-class** duplicate pairs (same image, different labels): **2**

## Shared-background pairs (compositing artefact)
- pairs sharing a background but not a target: **375**
- images involved: **365** (4.7%)
- of which **cross-class**: **10** pairs -- the same background plate carries targets of different types, i.e. the imagery is at least partly composited rather than natively cropped

## Leakage under a naive (non group-aware) stratified split
- held-out size (val + test): 3117 (ratios val 0.2, test 0.2)
- train/held-out duplicate pairs: **2334**
- held-out images having a twin in train: **1240**
- **leakage rate: 39.78% of held-out**

verdict: **LEAKAGE > 5% -- val accuracy is inflated, flag in REPORT.md**

## Residual leakage of the GROUP-AWARE split (threshold sweep)

Split as actually trained on: held-out (val + test) size 3040. Grouping is built at Hamming <= 5, so 0% at that threshold is true by construction. Loosening the threshold asks whether near-duplicates slipped through. Once `cross-class` climbs, the threshold has stopped meaning "duplicate" and the row is an upper bound, not a measurement.

| Hamming <= | train/held-out pairs | held-out images with a train twin | % of held-out | of which cross-class |
|---:|---:|---:|---:|---:|
| 5 | 0 | 0 | 0.00% | 0 |
| 8 | 116 | 53 | 1.74% | 2 |
| 10 | 215 | 98 | 3.22% | 6 |
| 12 | 436 | 199 | 6.55% | 33 |
| 16 | 2369 | 838 | 27.57% | 941 |

## Cross-class duplicate pairs (label-quality issue)

- `011.Asagiri-class_destroyer/P5985.bmp`  ==  `027.Murasame-class_destroyer/P9862.bmp`
- `011.Asagiri-class_destroyer/P6795.bmp`  ==  `027.Murasame-class_destroyer/P8782.bmp`