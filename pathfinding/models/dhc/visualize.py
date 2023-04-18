import fire
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import torch

from pathfinding.environment import Environment, MovingAIBenchmarkingEnvironment
from pathfinding.models.dhc import DHCNetwork
from pathfinding.utils import tests_dir_path

torch.manual_seed(239)
np.random.seed(239)
random.seed(239)
device = torch.device("cpu")
torch.set_num_threads(1)


def get_cmap(n, name="hsv"):
    return plt.cm.get_cmap(name, n)


def frametamer(imgs, env, init_img):
    imgs.append([])
    imgs[-1].append(init_img)

    num_agents = len(env.agents_pos)
    eq = env.agents_pos == env.goals_pos
    total_positioned = (eq[:, 0] * eq[:, 1]).sum()

    for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(
        zip(env.agents_pos, env.goals_pos)
    ):
        imgs[-1].append(
            plt.text(0.02, 0.02, s=f"{total_positioned} / {num_agents}", fontsize=8)
        )
        imgs[-1].append(
            plt.text(
                agent_y, agent_x, i, color="black", ha="center", va="center", fontsize=8
            )
        )
        imgs[-1].append(
            plt.text(
                goal_y, goal_x, i, color="black", ha="center", va="center", fontsize=8
            )
        )


def fill_map(env):
    map = np.copy(env.map)
    for agent_id in range(env.num_agents):
        x, y = env.agents_pos[agent_id], env.goals_pos[agent_id]
        if np.array_equal(x, y):
            map[tuple(x)] = 4
        else:
            map[tuple(x)] = 2
            map[tuple(y)] = 3
    map = map.astype(np.uint8)
    return map


def make_animation_single_text(
    model_id: int, test_name: str, test_case_idx: int = 0, steps: int = 256
):
    test_case_idx = int(test_case_idx)
    color_map = np.array(
        [
            [255, 255, 255],  # white
            [190, 190, 190],  # gray
            [0, 191, 255],  # blue
            [255, 165, 0],  # orange
            [0, 250, 154],  # green
        ]
    )

    network = DHCNetwork()
    network.eval()
    network.to(device)
    state_dict = torch.load(
        os.path.join(".", "models", f"{model_id}.pth"), map_location=device
    )
    network.load_state_dict(state_dict)

    with open(os.path.join(tests_dir_path(), test_name), "rb") as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure(figsize=(4.8, 4.8))
    plt.gca().set_xticks(range(0, len(env.map) + 1, 5))
    plt.gca().set_yticks(range(0, len(env.map) + 1, 5))

    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        map = fill_map(env)
        img = plt.imshow(color_map[map], animated=True)

        frametamer(imgs, env, img)

        actions, _, _, _ = network.step(
            torch.from_numpy(obs.astype(np.float32)).to(device),
            torch.from_numpy(pos.astype(np.float32)).to(device),
        )
        (obs, pos), _, done, _ = env.step(actions)

    if done and env.steps < steps:
        map = fill_map(env)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            frametamer(imgs, env, img)

    ani = animation.ArtistAnimation(
        fig, imgs, interval=600, blit=True, repeat_delay=1000
    )

    video_writer = animation.PillowWriter(fps=10)

    videos_dir = os.path.join(".", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    ani.save(
        os.path.join(videos_dir, f"test_{model_id}_{test_name}_{test_case_idx}.gif"),
        writer=video_writer,
    )


def make_animation_movingai(
    model_id: int, test_name: str, test_case_idx: int = 0, steps: int = 256
):
    test_case_idx = int(test_case_idx)
    color_map = np.array(
        [
            [255, 255, 255],  # white
            [190, 190, 190],  # gray
            [0, 191, 255],  # blue
            [255, 165, 0],  # orange
            [0, 250, 154],  # green
        ]
    )

    network = DHCNetwork()
    network.eval()
    network.to(device)
    state_dict = torch.load(
        os.path.join(".", "models", f"{model_id}.pth"), map_location=device
    )
    network.load_state_dict(state_dict)

    with open(os.path.join(tests_dir_path(), test_name), "rb") as f:
        tests = pickle.load(f)

    env = MovingAIBenchmarkingEnvironment(should_init=False)
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

    fig = plt.figure(figsize=(4.8, 4.8))

    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        map = fill_map(env)
        img = plt.imshow(color_map[map], animated=True)

        frametamer(imgs, env, img)

        actions, _, _, _ = network.step(
            torch.from_numpy(obs.astype(np.float32)).to(device),
            torch.from_numpy(pos.astype(np.float32)).to(device),
        )
        (obs, pos), _, done, _ = env.step(actions)

    if done and env.steps < steps:
        map = fill_map(env)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            frametamer(imgs, env, img)

    ani = animation.ArtistAnimation(
        fig, imgs, interval=600, blit=True, repeat_delay=1000
    )

    video_writer = animation.PillowWriter(fps=10)

    videos_dir = os.path.join(".", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    ani.save(
        os.path.join(videos_dir, f"{model_id}_{test_name}_{test_case_idx}.gif"),
        writer=video_writer,
    )


def _make_single_map_image_for_report():
    env = Environment(num_agents=8, map_length=40, fix_density=0.4)
    parts = env._part
    num_comp = len(parts)
    print(num_comp)

    fig = plt.figure(figsize=(4.8, 4.8))  # noqa

    cmap = [
        [255, 255, 255],
        [233, 150, 122],
        [238, 232, 170],
        [152, 251, 152],
        [102, 205, 170],
        [135, 206, 235],
        [255, 182, 193],
        [222, 184, 135],
        [255, 239, 213],
        [240, 255, 240],
        [192, 192, 192],
        [100, 149, 237],
        [72, 61, 139],
        [240, 230, 140],
        [0, 100, 0],
        [143, 188, 143],
        [95, 158, 160],
        [221, 160, 221],
        [250, 235, 215],
        [160, 82, 45],
        [255, 240, 245],
        [245, 255, 250],
        [112, 128, 144],
        [220, 220, 220],
        [255, 127, 80],
        [255, 140, 0],
        [128, 128, 0],
        [124, 252, 0],
        [47, 79, 79],
    ]

    map = np.copy(env.map)
    for agent_id in range(env.num_agents):
        x, y = env.agents_pos[agent_id], env.goals_pos[agent_id]
        if np.array_equal(x, y):
            map[tuple(x)] = 4
        else:
            map[tuple(x)] = 0
            map[tuple(y)] = 0

            plt.plot(
                x[1],
                x[0],
                marker="o",
                markersize=8,
                markerfacecolor="blue",
                markeredgecolor="blue",
            )
            plt.text(x[1] - 0.5, x[0] + 0.5, agent_id, fontsize=8, color="white")
            plt.plot(
                y[1],
                y[0],
                marker="o",
                markersize=8,
                markerfacecolor="orange",
                markeredgecolor="orange",
            )
            plt.text(y[1] - 0.5, y[0] + 0.5, agent_id, fontsize=8)

    map = map.astype(np.uint8)

    color_map = np.array(
        [
            [224, 255, 255],
            [190, 190, 190],  # gray
            [0, 191, 255],  # blue
            [255, 165, 0],  # orange
            [0, 250, 154],  # green
        ]
    )

    image = color_map[map]

    for i, c in enumerate(parts):
        color = np.array(cmap[i])
        for x, y in c:
            image[x, y] = color

    plt.imshow(image)
    plt.savefig("agents.png")


if __name__ == "__main__":
    fire.Fire()
