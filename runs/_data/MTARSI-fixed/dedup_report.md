# Duplicate / leakage report: MTARSI-fixed

threshold: Hamming <= 5 on **both** aHash and pHash, over variants ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip'], at two scales (full frame and centre 60% crop).

A full-frame match with a **non**-matching centre crop means the two images share a background but hold different targets -- a compositing artefact, not a duplicate. Those are counted separately below and are **not** merged into duplicate groups.

- images scanned: **2264**
- duplicate pairs found: **226**
- images belonging to a duplicate group: **371** (16.4%)
- unique images after grouping: **2062**
- largest duplicate group: **6** images
- groups with >1 member: 169
- **cross-class** duplicate pairs (same image, different labels): **0**

## Shared-background pairs (compositing artefact)
- pairs sharing a background but not a target: **63**
- images involved: **100** (4.4%)
- of which **cross-class**: **6** pairs -- the same background plate carries targets of different types, i.e. the imagery is at least partly composited rather than natively cropped

## Leakage under a naive (non group-aware) stratified split
- val size: 565 (ratio 0.25)
- cross-split duplicate pairs: **92**
- val images having a twin in train: **78**
- **leakage rate: 13.81% of val**

verdict: **LEAKAGE > 5% -- val accuracy is inflated, flag in REPORT.md**