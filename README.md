# LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences

<h2 style="text-align: center;">CVPR 2025
<a href="https://www.easycounter.com/"><img src="https://www.easycounter.com/counter.php?hoyyyaard" border="0" alt="Hit Counter"></a> <br><a href="https://www.easycounter.com/"></a>
</h2>

[[📖 arXiv]](https://arxiv.org/abs/2412.01292) 
[[🤖 model]](https://huggingface.co/Hoyard/LSceneLLM)
[[📑 dataset]](https://huggingface.co/datasets/Hoyard/XR-Scene)


This repository contains PyTorch implementation for __LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences__ 

In 3D scenes, visual information is often complex and abundant, especially in cross-room scenes and outdoor scenes. We propose a solution that reduces computational load while preserving detailed information by using the attention map of LLM to select tokens of interest, effectively integrating both coarse-grained and fine-grained visual information, and a cross-room 3D large scene understanding benchmark.

## 🔥News
✅「**2025-03-28**」 Training demo code released. \
✅「**2025-02-27**」 LSceneLLM is accepted by CVPR 2025 ! \
✅「**2025-01-31**」 Inference code, Pretrained weight, Annotation of XR-Scene released.



## 🔧Usage

### Requirements

- PyTorch >= 1.7.0
- python == 3.7
- CUDA >= 10.2
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset
#### Preprocess HM3D-OpenScene Features
- Install [OpenScene](https://github.com/pengsongyou/openscene) requirement
- Download HM3D scan data from [SceneVerse](https://scene-verse.github.io/) and put it into *data/SceneVerse/HM3D/[qa, caption, planning]*
- Simply run:
```
export PROJ_DIR=<your path to lscenellm project>
bash scripts/preprocess_openscene_fts.sh
```

### Evaluation of XR-QA
To eval the pretrained model on XR-QA, simply run: 
```
bash scripts/slurm.sh # In cluster
bash scripts/eval.sh
```

### Training of XR-QA
We trained out LSceneLLM on LL3DA datas and XR-Scene. User can process any 3D datasets based on Preprocess HM3D-OpenScene Features section. We provide demo code for training on XR-QA, simply run: 
```
bash scripts/slurm.sh # In cluster
bash scripts/train.sh
```

## License
MIT License

## Citation
If you find our work useful in your research, please consider citing: 
```
@article{zhi2024lscenellm,
  title={LSceneLLM: Enhancing Large 3D Scene Understanding Using Adaptive Visual Preferences},
  author={Zhi, Hongyan and Chen, Peihao and Li, Junyan and Ma, Shuailei and Sun, Xinyu and Xiang, Tianhang and Lei, Yinjie and Tan, Mingkui and Gan, Chuang},
  journal={arXiv preprint arXiv:2412.01292},
  year={2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Hoyyyaard/LSceneLLM&type=Date)](https://star-history.com/#Hoyyyaard/LSceneLLM&Date)