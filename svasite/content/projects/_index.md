---
title: "Step-by-Step Guide"
date: 2025-05-21
weight: 1
---
# Step-by-Step Guide

So, you've some histological images and a good dose of scientific curiosity. Now what? Let's walk through the pipeline together, breaking it down step by step.

1. **Prepare Your Data**:
   - 1.1 Stain the histological images of interest for CD31
   - 1.2 Ensure the images are formatted correctly
   - 1.3 Remove image artifacts
2. **Segment Vessels and Extract Features in Python**:
   - 2.1 Convertion of RGB to HSV format
   - 2.2 Application of Otsu’s thresholding
   - 2.3 Morphological operations
   - 2.4 Feature extraction
3. **Statistical analyses**:
   - 3.1 Normalization
4. **Infer drug distribution**
   - 4.1 Generation of synthetic vessel masks
   - 4.2 Definition of a known drug distribution function
   - 4.3 Application of GP modelling on simulator
   - 4.4 Assesment of model accuracy
5. **Interpret Your Data**