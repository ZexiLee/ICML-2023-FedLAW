# ICML-2023-FedLAW

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.02913-b31b1b.svg)](https://arxiv.org/pdf/2302.10911.pdf)

>**This is the official implementation of the ICML 2023 paper "Revisiting Weighted Aggregation in Federated Learning with Neural Networks".**

## Paper Overview
**TLDR**: We gain insights into the weighted aggregation of federated learning from global weight shrinking and client coherence. We also devise an effective aggregation algorithm FedLAW.

**Abstract**: In federated learning (FL), weighted aggregation of local models is conducted to generate a global model, and the aggregation weights are normalized (the sum of weights is 1) and proportional to the local data sizes. In this paper, we revisit the weighted aggregation process and gain new insights into the training dynamics of FL. First, we find that the sum of weights can be smaller than 1, causing _**global weight shrinking**_ effect (analogous to weight decay) and improving generalization. We explore how the optimal shrinking factor is affected by clients' data heterogeneity and local epochs. Second, we dive into the relative aggregation weights among clients to depict the clients' importance. We develop _**client coherence**_ to study the learning dynamics and find a critical point that exists. Before entering the critical point, more coherent clients play more essential roles in generalization. 
Based on the above insights, we propose an effective method for **Fed**erated Learning with **L**earnable **A**ggregation **W**eights, named **FedLAW**. Extensive experiments verify that our method can improve the generalization of the global model by a large margin on different datasets and models.

![image](https://github.com/ZexiLee/ICML-2023-FedLAW/blob/main/figs/fig1_2.png)

---
![image](https://github.com/ZexiLee/ICML-2023-FedLAW/blob/main/figs/fig5.png)


---
## Codebase Overview
- We implement our proposed FedLAW and the FL baselines (FedAvg, FedDF, FedBE, FedDyn, FedAdam, FedProx, and Server-finetune) into the same framework. We note that:
  - Running ''_main_fedlaw.py_'' for our proposed FedLAW.
  - Running ''_main_baselines.py_'' for all the baselines.
- We decouple the algorithms into the ''_server_method_'' and the ''_client_method_''. Therefore, the server and client methods can be flexibly combined. The original algorithms can be reproduced by setting:
  ```
  FedLAW: {'server_method': 'fedlaw', 'client_method': 'local_train'}.
  FedAvg: {'server_method': 'fedavg', 'client_method': 'local_train'}.
  FedDF: {'server_method': 'feddf', 'client_method': 'local_train'}.
  FedBE: {'server_method': 'fedlbe', 'client_method': 'local_train'}.
  FedDyn: {'server_method': 'feddyn', 'client_method': 'feddyn'}.
  FedAdam: {'server_method': 'fedadam', 'client_method': 'local_train'}.
  FedProx: {'server_method': 'fedavg', 'client_method': 'fedprox'}.
  Server-finetune: {'server_method': 'finetune', 'client_method': 'local_train'}.
  ```
  
## Citing This Repository

Please cite our paper if you find this repo useful in your work:

```
@InProceedings{pmlr-v202-li23s,
  title = 	 {Revisiting Weighted Aggregation in Federated Learning with Neural Networks},
  author =       {Li, Zexi and Lin, Tao and Shang, Xinyi and Wu, Chao},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {19767--19788},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/li23s/li23s.pdf},
  url = 	 {https://proceedings.mlr.press/v202/li23s.html},
  abstract = 	 {In federated learning (FL), weighted aggregation of local models is conducted to generate a global model, and the aggregation weights are normalized (the sum of weights is 1) and proportional to the local data sizes. In this paper, we revisit the weighted aggregation process and gain new insights into the training dynamics of FL. First, we find that the sum of weights can be smaller than 1, causing global weight shrinking effect (analogous to weight decay) and improving generalization. We explore how the optimal shrinking factor is affected by clients’ data heterogeneity and local epochs. Second, we dive into the relative aggregation weights among clients to depict the clients’ importance. We develop client coherence to study the learning dynamics and find a critical point that exists. Before entering the critical point, more coherent clients play more essential roles in generalization. Based on the above insights, we propose an effective method for Federated Learning with Learnable Aggregation Weights, named as FedLAW. Extensive experiments verify that our method can improve the generalization of the global model by a large margin on different datasets and models.}
}
```

## Contact

Please feel free to contact via email (<zexi.li@zju.edu.cn>) if you have further questions.
