# LSDF
This is the corresponding code implementation for Algorithm LSDF (Label-Specific Disentanglement and correlation-guided Fusion)
## Abstract
In multi-label classification (MLC), label-specific features have stronger connection with the label than original ones. Popular methods rely on either representative instance based transformation or shared feature selection. However, the former hardly achieves good coupling between stages, while the latter lacks pertinence in feature generation. In this paper, we propose an end-to-end approach named Label-Specific Disentanglement and correlation-guided Fusion (**LSDF**) to handle these issues. First, the self-attention mechanism is employed to explore shared feature representations with rich semantic information. Second, the specific features are disentangled from the share one via constructing dedicated multi-layer perceptron (MLPs). Third, guided by third-order label correlation, the tailored features of each label are enriched by fusing with that of the relevant two. Finally, the specific features of each label are separately fed into their respective classifiers for prediction. The results of comprehensive experiments on **16** benchmark datasets validate the superiority of LSDF against other well-established MLC algorithms.

![LSDF_Overview](https://github.com/FanSmale/LSDF/assets/30546261/d94ff5a3-6179-4ae9-8472-36bb306d1e12)

- Shared feature learning module aims to obtain well representations through exploring structured information among features.
- Label-specific disentanglement module generates tailored representations for each label through q independent MLPs.
- Correlation-guided fusion module assigns label correlation semantics to the customized features via graph isomorphic network.
## code structure
- main: the main experiment file with run example!
- model
  - LSDFmodel: the LSDF network.
  - network:   the network components that make up LSDFmodel.
- loss
  - loss: classification loss
- optimizer
  - optimization: the optimizer that defines the network optimization process.
- config
  - checkpoints: store the checkpoints for each dataset.
  - config:      we give the detailed configuration information of all **16** datasets in this file.
- results: store the experiment results for each dataset.
- utils
  - metric:     define the evaluation metrics that we used.
  - myutils:    some auxiliary functions.
  - properties: manage configuration information.
