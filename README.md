# [Re] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization
The official codes of our ML Reproducibility Challenge 2022 report: [[Re] Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization]()

Original Paper: [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)

**Summary:** In this reproducibility study, we present our results and experience during replicating the paper, titled Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization. In real‐world scenarios, the feature distributions are mostly much more complicated than Gaussian, so only mean and standard deviation may not be fully representative to match them. This paper introduces a novel strategy to exactly match the histograms of image features via the Sort‐Matching algorithm in a computationally feasible way. We were able to reproduce most of the results presented in the original paper both qualitatively and quantitatively. 


**Scope of Reproducibility** In the scope of this study, we aim to reproduce all the qualitative and quantitative results on two tasks, namely Arbitrary Style Transfer (AST) and Domain Generalization (DG). Moreover, we investigate the capability of forming better style representations by EFDM in another recent study.

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



In our report, We have reproduced the experiments done on two selected tasks, and compared their results with the reported results. Although our experimental results are not identical to the reported ones, we can validate the claims made by the original study according to these results. The source code for reproducing all experiments can be found in `EFDM/ArbitraryStyleTransfer`, `EFDM/DomainGeneralization/imcls`, and `EFDM/DomainGeneralization/reid`, respectively. Our reproduced results can be found in `EFDM/ArbitraryStyleTransfer/reproduced_results` and `EFDM/DomainGeneralization/reproduced_results`.

README file of the original paper are given in `OriginalPaper/README.md`.

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