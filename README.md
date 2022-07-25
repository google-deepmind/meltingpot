# Melting Pot

*A suite of test scenarios for multi-agent reinforcement learning.*


[![meltingpot-tests](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

<div align="center">
  <img src="docs/images/meltingpot_montage_360.gif"
       alt="Melting Pot substrates" />
</div>

## About

Melting Pot assesses generalization to novel social situations involving both
familiar and unfamiliar individuals, and has been designed to test a broad range
of social interactions such as: cooperation, competition, deception,
reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a
set of 21 multi-agent reinforcement learning _substrates_ (multi-agent games) on
which to train agents, and over 85 unique test _scenarios_ on which to evaluate
these trained agents. The performance of agents on these held-out test scenarios
quantifies whether agents:

*   perform well across a range of social situations where individuals are
    interdependent,
*   interact effectively with unfamiliar individuals not seen during training
*   pass a universalization test: answering positively to the question: _what
    if everyone behaved like that?_

The resulting score can then be used to rank different multi-agent RL algorithms
by their ability to generalize to novel social situations.

We hope Melting Pot will become a standard benchmark for multi-agent
reinforcement learning. We plan to maintain it, and will be extending it in the
coming years to cover more social interactions and generalization scenarios.

If you are interested in extending Melting Pot, please refer to the
[Extending Melting Pot](docs/extending.md) documentation.

## Installation

Melting Pot is built on top of
[DeepMind Lab2D](https://github.com/deepmind/lab2d). 

### Devcontainer (x86 only)

*NOTE: This Devcontainer only works for x86 platforms. For arm64 (newer M1 Macs) users will have to follow the manual installation steps.*

This project includes a pre-configured development environment ([devcontainer](https://containers.dev)).

You can launch a working development environment with one click, using e.g. [Github
Codespaces](https://github.com/features/codespaces) or the [VSCode
Containers](https://code.visualstudio.com/docs/remote/containers-tutorial) extension.

### Manual install

The installation steps are
as follows (see [`install.sh`](https://github.com/deepmind/meltingpot/blob/main/install.sh)
for an example installation script):

1.  (Optional) Activate a virtual environment, e.g.:

    ```shell
    python3 -m venv "${HOME}/meltingpot_venv"
    source "${HOME}/meltingpot_venv/bin/activate"
    ```

2.  Install `dmlab2d` from the
    [dmlab2d wheel files](https://github.com/deepmind/lab2d/releases/tag/release_candidate_2022-03-24), e.g.:

    ```shell
    pip3 install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl
    ```

    If there is no appropriate wheel (e.g. M1 chipset) you will need to install
    [`dmlab2d`](https://github.com/deepmind/lab2d) and build the wheel yourself
    (see
    [`install.sh`](https://github.com/deepmind/meltingpot/blob/main/install.sh)
    for an example installation script that can be adapted to your setup).

3.  Test the `dmlab2d` installation in `python3`:

    ```python
    import dmlab2d
    import dmlab2d.runfiles_helper

    lab = dmlab2d.Lab2d(dmlab2d.runfiles_helper.find(), {"levelName": "chase_eat"})
    env = dmlab2d.Environment(lab, ["WORLD.RGB"])
    env.step({})
    ```

4.  Install Melting Pot:

    ```shell
    git clone -b main https://github.com/deepmind/meltingpot
    cd meltingpot
    curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-1.0.0.tar.gz \
        | tar -xz --directory=meltingpot
    pip3 install .
    ```

5.  Test the Melting Pot installation:

    ```shell
    pip3 install pytest
    pytest meltingpot
    ```

## Example usage

You can try out the substrates interactively with the
[human_players](meltingpot/python/human_players) scripts. For example, to play the
`clean_up` substrate, you can run:

```shell
python3 meltingpot/python/human_players/play_clean_up.py
```

You can move around with the `W`, `A`, `S`, `D` keys, Turn with `Q`, and `E`,
fire the zapper with `1`, and fire the cleaning beam with `2`. You can switch
between players with `TAB`. There are other substrates available in the
[human_players](meltingpot/python/human_players) directory. Some have multiple variants,
which you select with the `--level_name` flag.

NOTE: If you get a `ModuleNotFoundError: No module named 'meltingpot.python'`
      error, you can solve it by exporting the meltingpot home directory as
      `PYTHONPATH` (e.g. by calling `export PYTHONPATH=$(pwd)`).

### Training agents

We provide two example scripts using RLlib and [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (SB3) respectively. Note that Melting Pot is agnostic to how you train your agents, and as such, these scripts are not meant to be a suggestion on how to achieve good scores in the task suite.

#### RLlib

[RLLib](https://github.com/ray-project/ray) is an open-source reinforcement
learning library, with support for distributed workloads.

The example trains multiple agents using a shared policy, with the
hyperparameters used in the original Melting Pot paper. It saves the trained
model as
[checkpoints](https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html)
in `examples/rllib/.results` and videos of gameplay to `examples/rllib/.videos`.

To use, install the specific requirements and then run the training script:

```shell
pip3 install .[rllib]
python ./examples/rllib/a3c.py
```

#### PettingZoo and Stable-Baselines3
This example uses a PettingZoo wrapper with a fully parameter shared PPO agent from SB3.

The PettingZoo wrapper can be used separately from SB3 and
can be found at [meltingpot_env.py](examples/pettingzoo/meltingpot_env.py)

```shell
cd <meltingpot_root>
pip3 install -e .[pettingzoo]
```

```shell
cd <meltingpot_root>/examples/pettingzoo
python3 sb3_train.py
```

### Documentation

Full documentation is available [here](docs/index.md).

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
    organization={PMLR}
}
```

## Disclaimer

This is not an officially supported Google product.
