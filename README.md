# Learnable Decentralized MAPF using reinforcement learning with local communication

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-%2300C4CC.svg?style=flat&logo=Poetry&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)

## Description

We perform extensive empirical evaluation of one of the state-of-the-art decentralized PO-MAPF algorithms which leverages communication between agents, Distributed Heuristic Communication (DHC). Through comprehensive experiments, the performance of DHC is observed to degrade when agents are faced with complete packet loss during communication. To mitigate this issue, we propose a novel algorithm called DHC-R (DHC-robust). Open-sourced model weights and the codebase are provided.

## Requirements
In order for `models.dhc.train` to be successfully run, you have to have a machine equipped with 1 GPU and several CPUs.
Consider having `num_cpus - 2` actors configured through the `dhc.train.num_actors` in `config.yaml`

**Attention: We do not guarantee the desired performance on a non-GPU machine.**

While we aim at supporting MacOS, Linux and Windows platforms, the successful training is not guaranteed on a Windows-based machine. 
The benchmarking script should work there, though. Please report it [here](https://github.com/acforvs/po-mapf-thesis/issues) if it doesn't.

## Setting up
1. Install [Poetry](https://python-poetry.org)
2. Run [poetry install](https://python-poetry.org/docs/cli/#install) to install the dependencies

If you see ``Failed to create the collection: Prompt dismissed..`` this error when trying to run `poetry install`, [consider](https://github.com/python-poetry/poetry/issues/1917#issuecomment-1251667047) executing this line first:
```shell
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

## Repository description & Usage 
1. `models` dir contains the weights of the trained models
2. `config.yaml` - training & model params, environmental settings etc.
3. `pathfinding/models` provides one with the implementation of different models


## Cite

```
@InProceedings{10.1007/978-3-031-43111-1_14,
author="Savinov, Vladislav
and Yakovlev, Konstantin",
editor="Ronzhin, Andrey
and Sadigov, Aminagha
and Meshcheryakov, Roman",
title="DHC-R: Evaluating ``Distributed Heuristic Communication'' and Improving Robustness for Learnable Decentralized PO-MAPF",
booktitle="Interactive Collaborative Robotics",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="151--163",
abstract="Multi-agent pathfinding (MAPF) is a problem of coordinating the movements of multiple agents operating a shared environment that has numerous industrial and research applications. In many practical cases the agents (robots) have limited visibility of the environment and must rely on local observations to make decisions. This scenario, known as partially observable MAPF (PO-MAPF), can be solved through decentralized approaches. In recent years, several learnable algorithms have been proposed for solving PO-MAPF. However, their performance is oftentimes not validated out-of-distribution (OOD), and the code is often not properly open-sourced. In this study, we conduct a comprehensive empirical evaluation of one of the state-of-the-art decentralized PO-MAPF algorithms, Distributed Heuristic Communication (DHC), Ma, Z., Luo, Y., Ma, H.: Distributed heuristic multi-agent path finding with communication. In: 2021 International Conference on Robotics and Automation (ICRA), pp. 8699--8705. IEEE, Xi'an, China (2021), which incorporates communication between agents. Our experiments reveal that the performance of DHC deteriorates when agents encounter complete packet loss during communication. To address this issue, we propose a novel algorithm called DHC-R that employs a similar architecture to the original DHC but introduces randomness into the graph neural network-based communication block, preventing the passage of some data packets during training. Empirical evaluation confirms that DHC-R outperforms DHC in scenarios with packet loss. Open-sourced model weights and the codebase are provided: https://github.com/acforvs/dhc-robust-mapf.",
isbn="978-3-031-43111-1"
}
```

## Contributing
<details>
    <summary>See the detailed contribution guide</summary>

1. Install [black](https://github.com/psf/black), you can likely run
```shell
pip3 install black 
```

2. Use [black](https://github.com/psf/black) to ensure that the codestyle remains great
```shell
poetry run black .
```
3. Use [ruff](https://github.com/charliermarsh/ruff) to lint all the files
```shell
poetry run ruff .
```
4. Make sure tests are OK 
```shell
poetry run pytest
```
5. Create a PR with new features
</details>

## References

<a id="1">[1]</a> 
Ma, Ziyuan and Luo, Yudong and Ma, Hang, 2021. Distributed Heuristic Multi-Agent Path Finding with Communication.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/acforvs/po-mapf-thesis/blob/main/LICENSE)


