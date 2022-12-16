## Attention mechanisms and Transformers

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Attention_and_Transformers) [![PyPI version](https://badge.fury.io/py/Attention-and-Transformers.svg)](https://badge.fury.io/py/Attention-and-Transformers) [![TensorFlow 2.10.0](https://img.shields.io/badge/TensorFlow-2.10\|2.11-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.10.0) [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)

* This goal of this repository is to host basic architecture and model traning code associated with the different attention mechanisms and transformer architecture.
* At the moment, I more interested in learning and recreating these new architectures from scratch than full-fledged training. For now, I'll just be training these models on small datasets.

#### Installation

* Using pip to install from [pypi](https://pypi.org/project/Attention-and-Transformers/)

```bash
pip install Attention-and-Transformers
```

* Using pip to install latest version from github

```bash
pip install git+https://github.com/veb-101/Attention-and-Transformers.git
```

* Local clone and install

```bash
git clone https://github.com/veb-101/Attention-and-Transformers.git atf
cd atf
python setup.py install
```

**Test Installation**

```bash
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
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/ViT/multihead_self_attention.py">Multi-head Self Attention</a>
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
<tr>
<td style="text-align:center">2</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/MobileViT_v2/linear_attention.py">Separable Self Attention</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2206.02680">MobileViT V2</a>
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
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/ViT/vision_transformer.py">Vision Transformer</a>
</td>
<td style="text-align:center">
<a href="https://arxiv.org/abs/2010.11929">An Image is Worth 16x16 Words:</a>
</td>
</tr>
<tr>
<td style="text-align:center">2</td>
<td style="text-align:center">
<a href="https://github.com/veb-101/Attention-and-Transformers/blob/main/Attention_and_Transformers/MobileViT_v1/mobile_vit_v1.py">MobileViT-V1</a>
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
