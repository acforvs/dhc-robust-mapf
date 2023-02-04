import fire
import numpy as np
import os
import pickle
import torch

from pathfinding.environment import Environment, MovingAIBenchmarkingEnvironment
from pathfinding.models.dhc import DHCNetwork
from pathfinding.settings import yaml_data as settings
from pathfinding.utils import test_group, calculate_metrics

GENERAL_CONFIG = settings["dhc"]


def _test_one_case(args):
    map, agents_pos, goals_pos, network, env_cls = args
    env = env_cls(should_init=False)
    env.load(map, agents_pos, goals_pos)
    obs, pos = env.observe()

    done, steps = False, 0
    network.reset()

    while not done and env.steps < 5 * GENERAL_CONFIG["max_episode_length"]:
        actions, _, _, _ = network.step(
            torch.as_tensor(obs.astype(np.float32)),
            torch.as_tensor(pos.astype(np.float32)),
        )
        (obs, pos), _, done, _ = env.step(actions)
        steps += 1

    return calculate_metrics(env, steps)


def _test_generation_fn_random(tests, network):
    return [(*test, network, Environment) for test in tests]


def _test_generation_fn_moving_ai(tests, network):
    return [(*test, network, MovingAIBenchmarkingEnvironment) for test in tests]


def test_model(
    test_groups=[
        (40, 4, 0.3),
        (40, 8, 0.3),
        (40, 16, 0.3),
        (40, 32, 0.3),
        (40, 64, 0.3),
        (80, 4, 0.3),
        (80, 8, 0.3),
        (80, 16, 0.3),
        (80, 32, 0.3),
        (80, 64, 0.3),
    ],
    model_number=60000,
    is_random_maps: bool = True,
):
    network = DHCNetwork()
    network.eval()
    device = torch.device("cpu")
    network.to(device)
    state_dict = torch.load(
        os.path.join(".", "models", f"{model_number}.pth"), map_location=device
    )
    network.load_state_dict(state_dict)
    network.eval()
    network.share_memory()

    if is_random_maps:
        func = lambda x: _test_generation_fn_random(x, network)
    else:
        func = lambda x: _test_generation_fn_moving_ai(x, network)

    for group in test_groups:
        test_group(group, func, _test_one_case, is_random=is_random_maps)


if __name__ == "__main__":
    fire.Fire()
