# PDSDNet: For Chemical plant Pipeline Dual-head-Segmentation Network-based-on-DINOv3
This network is designed specifically for pipeline extraction within chemical industrial parks and can also be used for road extraction. Due to file size and weight limitations, the full code for this project, as well as the optimal training weights for PDSDNet and the DINOv3 weights, can be downloaded from this link: https://pan.baidu.com/s/1dBpeJGFPac6CpmOgzpJ0sQ?pwd=88xg. ⚠️⚠️⚠️Please note that if you download the DINOv3 weights from this link, you must provide proper citations to acknowledge the original authors of DINOv3!⚠️⚠️⚠️

## PDSDNet OverView
The overall network architecture of the method proposed in this paper (\ref{fig:fig2}) uses the DINOv3 ViT-L model, trained on the SAT493M dataset, as the backbone network. The model’s parameters are frozen, eliminating the need for fine-tuning, and it is directly employed as a hierarchical feature extractor. For an input image $I \in \mathbb{R}^{3 \times H \times W}$, we extract intermediate feature maps from the 7th, 15th, and 23rd Transformer layers of the ViT-L encoder, respectively. These multi-level representations simultaneously capture both local textural details and global semantic information. First, the features from the 23rd layer, $\mathbf{F}_{23} \in \mathbb{R}^{1024 \times \frac{H} {16} \times \frac{W}{16}}$ is projected via a $1\times 1$ convolution and spatially downsampled to form the base feature $\mathbf{p}_5 \in \mathbb{R}^{256 \times \frac{H}{32} \times \frac{W}{32}}$. Subsequently, through a series of additional $1\times 1$ convolutions and bilinear upsampling operations, a multi-scale feature pyramid $\{\mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4, \mathbf{p}_5\}$ with spatial dimensions of $\frac{H}{4}$, $\frac{H}{8}$, $\frac{H}{16}$, and $\frac{H}{32}$, respectively, and 256 channels each.
<img width="5631" height="2610" alt="framework" src="https://github.com/user-attachments/assets/f489ec40-1ead-4015-a4ce-7bdd66766b0d" />
To decode high-resolution structural details from this feature pyramid, this paper proposes a novel Pipeline Decoding Block (PDB). This module employs a component called StripDCN to perform hierarchical refinement and upsampling of features. Three PDB modules are cascaded to sequentially generate decoded features $\mathbf{d}_4 \in \mathbb{R}^{128 \times \frac{H} {8} \times \frac{W}{8}}$, $\mathbf{d}_3 \in \mathbb{R}^{64 \times \frac{H}{4} \times \frac{W}{4}}$, and $\mathbf{d}_2 \in \mathbb{R}^{32 \times \frac{H}{2} \times \frac{W}{2}}$. The final stage further outputs $\mathbf{d}_1 \in \mathbb{R}^{16 \times H \times W}$, which is then convolved with a $1\times 1$ kernel to obtain the coarse prediction map $\mathbf{d}_0 \in \mathbb{R}^{1 \times H \times W}$.
Furthermore, this paper introduces a Cascading Refinement Module (CRM), which explicitly integrates an edge mask to enhance the localization accuracy of object boundaries in the prediction map. The CRM iteratively fuses the coarse prediction $\mathbf{d}_0$, the edge-aware mask, and the high-resolution features $\mathbf{d}_1$ to produce the final refined result $\mathbf{O} \in \mathbb{R}^{1 \times H \times W}$.

## Pipeline Decoder Block (PDB)
The Pipeline Decoder module (as shown below) is designed to progressively up-sample and enhance multi-scale backbone features while preserving fine-scale spatial details to the greatest extent possible. Each PDB module takes two inputs: a lower-resolution feature map $\mathbf{p}_i$ from the feature pyramid, and an output feature map $\mathbf{d}_i+1$ from the previous decoding stage. The module is responsible for generating a higher-resolution feature map $\mathbf{d}_i$.
<img width="3861" height="2649" alt="PDB" src="https://github.com/user-attachments/assets/ebaf9dfe-4af7-4ce6-8a58-bd902a0d4772" />

## Cascading Refinement Module (CRM)
Although the coarse prediction map $\mathbf{d}_0 \in \mathbb{R}^{1 \times H \times W}$ captures the general structure of the object, its boundaries typically lack precise alignment and sharpness. To address this, we propose the Cascading Refinement Module (CRM), which explicitly utilizes edge information to iteratively refine the prediction results. At the core of the CRM is a shared-weight refinement head, whose structure is shown as below.
<img width="3516" height="1364" alt="CRM" src="https://github.com/user-attachments/assets/ef93df19-7b24-4966-b3ef-d3ae29457e29" />

## Training PDSDNet
Set the hyperparameters and dataset paths in train.py, then click “Run.” A loss curve and accuracy curve will be saved every 5 epochs. Once training is complete, two parameter files will be saved: best_model.pth and last_model.pth.The figure below shows the loss function curve for this experiment.
<img width="3566" height="1767" alt="training_curve_epoch_100" src="https://github.com/user-attachments/assets/5c6aa32a-7230-4c26-a716-97a079a40d8b" />


## Testing PDSDNet
Set the parameters, save path, and paths to the training and test datasets in test.py, then click Run. Once the test is complete, the console will display the test metrics (including Accuracy, Precision, Recall, F1-Score, FIOU, and MIOU). The final data is preserved as follows:

Save path/masks_tif: Inference results for the test set(The inference result is a binary image with values of 0 and 255)

Examples：[MeishanGF2cj_tile_00873.tif](https://github.com/user-attachments/files/27039347/MeishanGF2cj_tile_00873.tif),
[NingboGF7cj_tile_08540.tif](https://github.com/user-attachments/files/27039342/NingboGF7cj_tile_08540.tif),
[WeifangcjGF2_tile_00818.tif](https://github.com/user-attachments/files/27039316/WeifangcjGF2_tile_00818.tif),
[YantaiGF2cj_tile_01213.tif](https://github.com/user-attachments/files/27039303/YantaiGF2cj_tile_01213.tif).

## Evaluate
Use `evaluate.py` to perform a comprehensive accuracy evaluation of the inference results. Simply set the test set path and the output path for the inference results, then click “Run.” In addition to the accuracy score, the console will also display the ALPS, Connectivity, and Completeness metrics.

## Experimental Results
Figure shows the pipeline extraction results of PDSDNet in typical local areas of the four chemical industrial parks, where the blue regions represent the pipeline masks predicted by the model.




## Data Availability
Unfortunately, due to project and research group requirements, we cannot make the entire dataset publicly available; however, we will still provide a portion of the dataset for reference. The download link is: https://pan.baidu.com/s/1GMqBjMAb71xKUCBhnvGo7Q?pwd=ttkk

## ⚠️⚠️⚠️Notes⚠️⚠️⚠️
Please note that this experiment uses pre-trained weights from Meta’s DINOv3, which were trained on the SAT493M dataset. If you use the network built in this experiment or download the DINOv3 weights from here, please be sure to cite the source：
```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```

## Citing PDSDNet
If you find this repository useful, please consider giving a star :star:~ and citing :t-rex: 🐁:
'''
The paper has not yet been published.....TAT
'''
