# Melting Pot Examples

This directory contains example code showing how to use Melting Pot with various reinforcement learning frameworks.

## Examples

- **gym**: Example showing how to use Melting Pot with the Gymnasium API
- **pettingzoo**: Example showing how to use Melting Pot with PettingZoo and Stable Baselines 3
- **rllib**: Example showing how to use Melting Pot with Ray RLlib
- **tutorial**: Tutorial examples for learning how to use Melting Pot

## Requirements

Each example has its own dedicated `requirements.in` file containing the specific dependencies needed for that example. This allows you to install only the dependencies required for the specific example you want to run.

To install dependencies for a specific example:

```bash
# Install dependencies for a specific example (e.g., rllib)
pip install -r examples/rllib/requirements.in

# Run the example
python -m examples.rllib.self_play_train
```
