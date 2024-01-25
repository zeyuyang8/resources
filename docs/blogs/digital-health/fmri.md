---
title: Processing fMRI Data
layout: default
parent: Digital Health
grand_parent: Blogs
---

# Processing fMRI Data

## Helpful Links

- [Resting-state functional connectivity](https://www.youtube.com/playlist?list=PL0ka7t7wbhYiM0nn-nZpw5AQA3cvytNik)
- [Nilearn](https://nilearn.github.io/dev/index.html)
- [MRI analysis in Python using Nipype, Nilearn and more](https://peerherholz.github.io/workshop_weizmann/index.html)

## Data Processing

- Analyze every voxel in the brain
- Analyze atlas regions (reduce dimensionality)
  - Use average of voxels in each region
- Network analysis (not good for high-dimensional data)

## Analysis

- Consider as time series (`n_atlas_regions` x `n_time_points`)
- Connetivity analysis with prior expert knowledge
  - Correlation between atlas regions
- Independent component analysis
  - Find independent components
  - Find correlation between components
- Graph analysis
