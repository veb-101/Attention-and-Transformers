## Attention mechanisms and Transformers

[![Python 3.10.4](https://img.shields.io/badge/Python-3.10.4-3776AB)](https://www.python.org/downloads/release/python-3104/) [![TensorFlow 2.10.0](https://img.shields.io/badge/TensorFlow-2.10.0-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.10.0) [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

* This goal of this repository is to host basic architecture and model traning code associated with the different attention mechanisms and transformer architecture.
* At the moment, I more interested in learning and recreating these new architectures from scratch than full-fledged training. For now, I'll just be training these models on small datasets.

#### Setup

**Installation**

```bash
pip install Attention-and-Transformers==0.0.1
```

**Test Installation**

```
python load_test.py
```

**Attention Mechanisms**

<table>
<thead>
<tr>
<th style="text-align:center">
<strong># No.</strong>
</th>
<th style="text-align:center">
<strong>Mechanism</strong>
</th>
<th style="text-align:center">
<strong>Paper</strong>
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">1</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/VisionTransformers/multihead_self_attention.py">Multi-head Self Attention</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/1706.03762">Attention is all you need</a>
</td>
</tr>
<tr>
<td style="text-align:center">2</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/MobileViT_v1/multihead_self_attention_2D.py">Multi-head Self Attention 2D</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2110.02178">MobileViT V1</a>
</td>
</tr>
</tbody>
</table>

**Transformer Models**

<table>
<thead>
<tr>
<th style="text-align:center">
<strong># No.</strong>
</th>
<th style="text-align:center">
<strong>Models</strong>
</th>
<th style="text-align:center">
<strong>Paper</strong>
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">1</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/VisionTransformers/vision_transformer.py">Vision Transformer</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words:</a>
</td>
</tr>
<tr>
<td style="text-align:center">2</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/MobileViT_v1/mobile_vit.py">MobileViT-V1</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2110.02178">MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer</a>
</td>
</tr>
<tr>
<td style="text-align:center">3</td>
<td style="text-align:center">MobileViT-V2 (under development)</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2206.02680">Separable Self-attention for Mobile Vision Transformers</a>
</td>
</tr>
</tbody>
</table>

<!-- **Attention Mechanisms**

|:---------:|:----------------------------:|:-------------------------------------------------------------:|
| 1         | [Multi-head Self Attention](https://github.com/veb-101/Attention-and-Transformers/blob/main/MobileViT-v1/multihead_self_attention_2D.py)    | [Attention is all you need](https://arxiv.org/abs/1706.03762) |
| 2         | [Multi-head Self Attention 2D](https://github.com/veb-101/Attention-and-Transformers/blob/main/MobileViT_v1/multihead_self_attention_2D.py) | [MobileViT V1](https://arxiv.org/abs/2110.02178)              |

**Transformer Models**

| **# No.** | **Models**         | **Paper**                                                          |
|:---------:|:------------------:|:------------------------------------------------------------------:|
| 1         | [Vision Transformer](https://github.com/veb-101/Attention-and-Transformers/blob/main/VisionTransformers/vision_transformer.py) | [An Image is Worth 16x16 Words:](https://arxiv.org/abs/2010.11929) |
| 2         | [MobileViT-V1](https://github.com/veb-101/Attention-and-Transformers/blob/main/MobileViT_v1/mobile_vit.py)     | [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178)                   |
| 3         | MobileViT-V2 (under development)| [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680)                   | -->
