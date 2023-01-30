# [Re] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization
The official codes of our ML Reproducibility Challenge 2022 report: [[Re] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization]()

Original Paper: [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)
by Yabin Zhang, Minghan Li, Ruihuang Li, Kui Jia, Lei Zhang.

Original Repository: [EFDM](https://github.com/YBZh/EFDM/) by [YBZh](https://github.com/YBZh/)

README file of the original paper are given in `OriginalPaper/README.md`. 

**Summary:** In this reproducibility study, we present our results and experience during replicating the paper, titled Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. In real‐world scenarios, the feature distributions are mostly much more complicated than Gaussian, so only mean and standard deviation may not be fully representative to match them. This paper introduces a novel strategy to exactly match the histograms of image features via the Sort‐Matching algorithm in a computationally feasible way. We were able to reproduce most of the results presented in the original paper both qualitatively and quantitatively. 


**Scope of Reproducibility:** In the scope of this study, we aim to reproduce all the qualitative and quantitative results on two tasks, namely Arbitrary Style Transfer (AST) and Domain Generalization (DG). Moreover, we investigate the capability of forming better style representations by EFDM in another recent study.

Below we show a brief implementation of it in PyTorch:
```python
import torch
def exact_feature_distribution_matching(content, style):
    assert (content.size() == style.size()) ## content and style features should share the same shape
    B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
    _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
    value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
    return transferred_content.view(B, C, W, H)
```

**Reproduction Results:**

* Visual comparison of the reported and our produced results on standard (the first two rows) and photo‐realistic (the last two rows) style transfer.

![][fig-1]

* Illustration of style interpolation between a single content image and four style images.

![][fig-3]

* Average running time of prior methods and proposed strategy used in AST on a 512px
image. Note that the compared methods run on a single Tesla V100, while our measurement has been done on a single RTX 2080Ti.

|   Method   | Gatys et al. |  CMD  |  HM  |  AdaIN  |  EFDM  |  EFDM (R) |
|   ------   | ------------ |  ---  |  --  |  -----  |  ----  |  -------- |
|  Time (s)  |     25.61    | 19.84 | 0.33 |  0.0038 | 0.0039 |  0.0043   |

* Illustration of the content‐style trade‐off with different λ values.

![][fig-2]

* Comparison of reproduced results of different feature distribution matching strategies applied in the original paper.

![][fig-4]

* Ablation on trade‐off between content and style loss terms.

![][fig-5]

* DG results of category classification on PACS. (R) refers to our reproduced results

| Method | Art | Cartoon | Photo | Sketch | Average |
| ------ | --- | ------- | ----- | ------ | ------- |
| Leave‐one‐domain‐out generalization |
| R‐18 w/ MixStyle | 83.1±0.8 | 78.6±0.9 | 95.9±0.4 | 74.2±2.7 | 82.9 |
| R‐18 w/ EFDMix | 83.9±0.4 | 79.4±0.7 | 96.8±0.4 | 75.0±0.7 | 83.9 |
| R‐18 w/ EFDMix (R) | 80.6±1.5 | 78.1±0.6 | 94.1±0.9 | 72.3±1.2 | 81.3 |
| R‐18 w/ EFDMix (R) α = 0.5 | 80.7±1.8 | 78.2±1.0 | 94.2±1.3 | 71.4±1.9 | 81.3 |
| R‐18 w/ EFDMix (R) α = 1.0 | 80.9±1.4 | 78.1±0.9 | 94.14±1.3 | 71.4±2.1 | 81.1 |
| R‐50 w/ MixStyle | 90.3±0.3 | 82.3±0.7 | 97.7±0.4 | 74.7±0.7 | 86.2 |
| R‐50 w/ EFDMix | 90.6±0.3 | 82.5±0.7 | 98.1±0.2 | 76.4±1.2 | 86.9 |
| R‐50 w/ EFDMix (R) | 87.4±1.6 | 81.8±1.6 | 94.3±2.2 | 73.7±1.7 | 84.3 |
| R‐50 w/ EFDMix (R) α = 0.5 | 87.6±1.7 | 81.1±1.3 | 94.5±1.7 | 73.9±1.5 | 84.3 |
| R‐50 w/ EFDMix (R) α = 1.0 | 87.4±2.1 | 81.6±1.4 | 94.7±1.6 | 74.3±1.6 | 84.5 |
| Single source generalization |
| R‐18 w/ MixStyle | 61.9±2.2 | 71.5±0.8 | 41.2±1.8 | 32.2±4.1 | 51.7 |
| R‐18 w/ EFDMix | 63.2±2.3 | 73.9±0.7 | 42.5±1.8 | 38.1±3.7 | 54.4 |
| R‐18 w/ EFDMix (R) | 63.5±3.4 | 72.9±1.2 | 41.9±1.4 | 36.3±3.1 | 53.7 |
| R‐18 w/ EFDMix (R) α = 0.5 | 63.8±2.4 | 73.2±0.9  | 42.5±1.6 | 37.1±3.0 | 54.2 |
| R‐18 w/ EFDMix (R) α = 1.0 | 63.7±3.4 | 73.2±0.9 | 41.9±1.75 | 36.3±2.4 | 53.7 |
| R‐50 w/ MixStyle | 73.2±1.1 | 74.8±1.1 | 46.0±2.0 | 40.6±2.0 | 58.6 |
| R‐50 w/ EFDMix | 75.3±0.9 | 77.4±0.8 | 48.0±0.9 | 44.2±2.4 | 61.2 |
| R‐50 w/ EFDMix (R) | 73.0±2.2 |77.2±0.9 | 48.3±1.2 | 47.7±2.7 | 61.6 |
| R‐50 w/ EFDMix (R) α = 0.5 | 73.8±1.6 | 77.6±1.3 | 47.9±1.2 | 46.7±3.2 | 61.5 |
| R‐50 w/ EFDMix (R) α = 1.0 | 73.7±1.2 | 77.8±0.4 | 47.9±0.7 | 46.0±4.2 | 61.4 | 

* DG results on person re‐ID task. (R) refers to our reproduced results.

| Methods | MarKet1501 → GRID | | | | GRID → MarKet1501 | | | |
| ------- | ----------------- | -- | -- | -- | ----------------- | -- | -- | --  |
|  | mAP | R1  | R5 |  R10 | mAP |  R1 | R5 |  R10 |
| OSNet + MixStyle | 33.8±0.9 | 24.89±1.6 | 43.7±2.0 | 53.1±1.6 | 4.9±0.2 | 15.4±1.2 | 28.4±1.3 | 35.7±0.9 |
| OSNet + EFDMix | 35.5±1.8 | 26.7±3.3 | 44.4±0.8 | 53.6±2.0 | 6.4±0.2 | 19.9±0.6 | 34.4±1.0 | 42.2±0.8 |
| OSNet + EFDMix (R) | 35.0±2.6 | 25.1±2.3 | 45.6±4.1 | 52.0±2.9 | 6.2±0.7 | 18.8±1.8 | 33.6±2.7 | 41.4±2.9 |

* White‐balance correction results of the recent methods and its variant with EFDM on mixed‐illuminant evaluation set.

| Method |   | MSE↓ |   |   |   | ∆E 2000↓  |   |   |
| ------ | - | ----- | - | - | - | ---------  | - | - |
|        | Mean | Q1 | Q2 | Q3 | Mean | Q1 | Q2 | Q3 |
| StyleWB | 822.77 | 572.52 | 840.67 | 1025.26 | 11.65 | 10.63 | 11.86 | 13.02 |
| SR + AdaIN | 818.99 | 527.34 | 875.56 | 1049.03 | 11.01 | 8.64 | 11.41 | 12.31 |
| SR + EFDM | 761.05 | 513.96 | 818.39 | 969.33 | 10.16 | 8.75 | 9.81 | 11.69 |

In our report, We have reproduced the experiments done on two selected tasks, and compared their results with the reported results. Although our experimental results are not identical to the reported ones, we can validate the claims made by the original study according to these results. The source code for reproducing all experiments can be found in `EFDM/ArbitraryStyleTransfer`, `EFDM/DomainGeneralization/imcls`, and `EFDM/DomainGeneralization/reid`, respectively. Our reproduced results can be found in `EFDM/ArbitraryStyleTransfer/reproduced_results` and `EFDM/DomainGeneralization/reproduced_results`.

**What was easy:** The given code in the original repository was easy to follow, and it was well‐written in general. The authors designed the documentation and the source code in a way that anyone who has fundamental knowledge of Python could run the experiments, or even generate their own stylized image from any content.

**What was difficult:** We would like to add the reproduced outputs by Histogram Matching (HM) along with the others, however the training of HM was based on CPU and the estimated time to complete a single training was around 15 days in our setup. Consequently, we could not include the reproduced outputs by HM to this report. Moreover, it could not be possible to add t‐SNE visualizations to this report, as in the original paper, due to the lack of clarity in the documentation of its script.

To cite the original paper and the reproduction paper in your publications, please use the following bibtex entries:
```
@inproceedings{zhang2021exact,
  title={Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization},
  author={Zhang, Yabin and Li, Minghan and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  booktitle={CVPR},
  year={2022}
}
```
```
```

[fig-1]: ArbitraryStyleTransfer/reproduced_results/figures/efdm_fig1.png
[fig-2]: ArbitraryStyleTransfer/reproduced_results/figures/efdm_fig2.png
[fig-3]: ArbitraryStyleTransfer/reproduced_results/figures/efdm_fig3.png
[fig-4]: ArbitraryStyleTransfer/reproduced_results/figures/efdm_fig4.png
[fig-5]: ArbitraryStyleTransfer/reproduced_results/figures/efdm_fig5.png