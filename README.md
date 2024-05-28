# Melting Pot

*A suite of test scenarios for multi-agent reinforcement learning.*

<!-- GITHUB -->
[![Python](https://img.shields.io/pypi/pyversions/dm-meltingpot.svg)](https://pypi.python.org/pypi/dm-meltingpot)
[![PyPI version](https://img.shields.io/pypi/v/dm-meltingpot.svg)](https://pypi.python.org/pypi/dm-meltingpot)
[![PyPI tests](../../actions/workflows/pypi-test.yml/badge.svg)](../../actions/workflows/pypi-test.yml)
[![Tests](../../actions/workflows/test-meltingpot.yml/badge.svg)](../../actions/workflows/test-meltingpot.yml)
[![Examples](../../actions/workflows/test-examples.yml/badge.svg)](../../actions/workflows/test-examples.yml)
<!-- /GITHUB -->

<div align="center">
  <img src="https://github.com/google-deepmind/meltingpot/blob/main/docs/images/meltingpot_montage.gif?raw=true"
       alt="Melting Pot substrates"
       height="250" width="250" />
</div>

[Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
[Melting Pot Contest at NeurIPS 2023](https://www.aicrowd.com/challenges/meltingpot-challenge-2023)

## About

Melting Pot assesses generalization to novel social situations involving both
familiar and unfamiliar individuals, and has been designed to test a broad range
of social interactions such as: cooperation, competition, deception,
reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a
set of over 50 multi-agent reinforcement learning _substrates_ (multi-agent
games) on which to train agents, and over 256 unique test _scenarios_ on which
to evaluate these trained agents. The performance of agents on these held-out
test scenarios quantifies whether agents:

*   perform well across a range of social situations where individuals are
    interdependent,
*   interact effectively with unfamiliar individuals not seen during training

The resulting score can then be used to rank different multi-agent RL algorithms
by their ability to generalize to novel social situations.

We hope Melting Pot will become a standard benchmark for multi-agent
reinforcement learning. We plan to maintain it, and will be extending it in the
coming years to cover more social interactions and generalization scenarios.

If you are interested in extending Melting Pot, please refer to the
[Extending Melting Pot](https://github.com/google-deepmind/meltingpot/blob/main/docs/extending.md) documentation.

## Installation

### `pip` install

[Melting Pot is available on PyPI](https://pypi.python.org/pypi/dm-meltingpot)
and can be installed using:

```shell
pip install dm-meltingpot
```

NOTE: Melting Pot is built on top of [DeepMind Lab2D](https://github.com/google-deepmind/lab2d)
which is distributed as pre-built wheels. If there is no appropriate wheel for
`dmlab2d`, you will need to build it from source (see
[the `dmlab2d` `README.md`](https://github.com/google-deepmind/lab2d/blob/main/README.md)
for details).

### Manual install

If you want to work on the Melting Pot source code, you can perform an editable
installation as follows:

1.  Clone Melting Pot:

    ```shell
    git clone -b main https://github.com/google-deepmind/meltingpot
    cd meltingpot
    ```

2.  (Optional) Activate a virtual environment, e.g.:

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install Melting Pot:

    ```shell
    pip install --editable .[dev]
    ```

4.  (Optional) Test the installation:

    ```shell
    pytest --pyargs meltingpot
    ```

### Devcontainer (x86 only)

*NOTE: This Devcontainer only works for x86 platforms. For arm64 (newer M1 Macs)
users will have to follow the manual installation steps.*

This project includes a pre-configured development environment
([devcontainer](https://containers.dev)).

You can launch a working development environment with one click, using e.g.
[Github Codespaces](https://github.com/features/codespaces) or the
[VSCode Containers](https://code.visualstudio.com/docs/remote/containers-tutorial)
extension.

#### CUDA support

To enable CUDA support (required for GPU training), make sure you have the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
package installed, and then run Docker with the `---gpus all` flag enabled. Note
that for GitHub Codespaces this isn't necessary, as it's done for you
automatically.

## Example usage

### Evaluation
The [evaluation](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/utils/evaluation/evaluation.py) library can be used
to evaluate [SavedModel](https://www.tensorflow.org/guide/saved_model)s
trained on Melting Pot substrates.

Evaluation results from the [Melting Pot 2.0 Tech Report](https://arxiv.org/abs/2211.13746)
can be viewed in the [Evaluation Notebook](https://github.com/google-deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/meltingpot/blob/main/notebooks/evaluation_results.ipynb)

### Interacting with the substrates

You can try out the substrates interactively with the
[human_players](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/human_players) scripts. For example, to play
the `clean_up` substrate, you can run:

```shell
python meltingpot/human_players/play_clean_up.py
```

You can move around with the `W`, `A`, `S`, `D` keys, Turn with `Q`, and `E`,
fire the zapper with `1`, and fire the cleaning beam with `2`. You can switch
between players with `TAB`. There are other substrates available in the
[human_players](https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/human_players) directory. Some have multiple
variants, which you select with the `--level_name` flag.

### Training agents

We provide two example scripts: one using
[RLlib](https://github.com/ray-project/ray), and another using
[PettingZoo](https://github.com/Farama-Foundation/PettingZoo) with
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3). Note
that Melting Pot is agnostic to how you train your agents, and as such, these
scripts are not meant to be a suggestion on how to achieve good scores in the
task suite.

#### RLlib

This example uses RLlib to train agents in
self-play on a Melting Pot substrate.

First you will need to install the dependencies needed by the examples:

```shell
cd <meltingpot_root>
pip install -r examples/requirements.txt
```

Then you can run the training experiment using:

```shell
cd examples/rllib
python self_play_train.py
```

#### PettingZoo and Stable-Baselines3

This example uses a PettingZoo wrapper with a fully parameter shared PPO agent
from SB3.

The PettingZoo wrapper can be used separately from SB3 and
can be found [here](https://github.com/google-deepmind/meltingpot/blob/main/examples/pettingzoo/utils.py).

```shell
cd <meltingpot_root>
pip install -r examples/requirements.txt
cd examples/pettingzoo
python sb3_train.py
```

## Documentation

Full documentation is available [here](https://github.com/google-deepmind/meltingpot/blob/main/docs/index.md).

## Citing Melting Pot

If you use Melting Pot in your work, please cite the accompanying article:

```bibtex
@inproceedings{leibo2021meltingpot,
    title={Scalable Evaluation of Multi-Agent Reinforcement Learning with
           Melting Pot},
    author={Joel Z. Leibo AND Edgar Du\'e\~nez-Guzm\'an AND Alexander Sasha
            Vezhnevets AND John P. Agapiou AND Peter Sunehag AND Raphael Koster
            AND Jayd Matyas AND Charles Beattie AND Igor Mordatch AND Thore
            Graepel},
    year={2021},
    journal={International conference on machine learning},
    organization={PMLR},
    url={https://doi.org/10.48550/arXiv.2107.06857},
    doi={10.48550/arXiv.2107.06857}
}
```

## Disclaimer

This is not an officially supported Google product.
