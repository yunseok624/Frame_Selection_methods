# Frame Selection Methods for Video LLMs

Welcome! This repository is dedicated to exploring and benchmarking various frame selection strategies for Video Language Models (Video LLMs), focusing on tasks like Video Reasoning and Video Question Answering (VQA).

## Abstract

## Results

<table>
  <thead>
    <tr>
      <th><div align="center"><br>MLLM</br></div></th>
      <th><div align="center"><br>Method</br></div></th>
      <th><div align="center"><br>Frames</br></div></th>
      <th><div align="center"><br>LLM param</br></div></th>
      <th><div align="center"><br>LVB</br></div></th>
      <th><div align="center"><br>V-MME</br></div></th>
      <th><div align="center"><br>MLVU</br></div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6" align="center"><b>Qwen2-VL</b></td>
      <td align="center">Uniform</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">AKS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FOCUS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">Q-Frame</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">MDP3</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FRAG</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><b>LLaVA-Video</b></td>
      <td align="center">Uniform</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">57.59</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">AKS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">60.21</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FOCUS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">Q-Frame</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">MDP3</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FRAG</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><b>LLaVA-OneVision</b></td>
      <td align="center">Uniform</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">AKS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FOCUS</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">Q-Frame</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">MDP3</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td align="center">FRAG</td>
      <td align="center">32</td>
      <td align="center">7B</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
  </tbody>
</table>

## Acknowledgment

This project is based on AKS ([paper](https://arxiv.org/abs/2502.21271), [code](https://github.com/ncTimTang/AKS)), FOCUS ([paper](https://arxiv.org/abs/2510.27280), [code](https://github.com/NUS-HPC-AI-Lab/FOCUS)), Q-Frame ([paper](https://arxiv.org/abs/2506.22139), [code](https://github.com/xiaomi-research/q-frame)), MDP3 ([paper](https://arxiv.org/abs/2501.02885), [code](https://github.com/sunh-23/MDP3)), FRAG ([paper](https://arxiv.org/abs/2504.17447), [code](https://github.com/NVlabs/FRAG)), LLaVA-NeXT ([paper](https://arxiv.org/abs/2410.02713), [code](https://github.com/LLaVA-VL/LLaVA-NeXT)), lmms_eval([paper](https://arxiv.org/abs/2407.12772), [code](https://github.com/EvolvingLMMs-Lab/lmms-eval))