import fire
import multiprocessing as mp
import numpy as np
import os
import pickle

from pathfinding.environment import Environment, MovingAIBenchmarkingEnvironment
from pathfinding.movingai import (
    get_tests_for_multiple_agents,
    read_map,
    TestDescription,
)


def generate_test_filename(length: int, num_agents: int, density: float, ext="pkl"):
    return f"{length}length_{num_agents}agents_{density}density.{ext}"


def generate_moving_ai_test_filename(
    map_filename: str, scenary_filename: str, num_agents: int, ext="pkl"
):
    map_filename = map_filename.split(os.sep)[-1]
    scenary_filename = scenary_filename.split(os.sep)[-1]
    return f"{map_filename}_{scenary_filename}_{num_agents}agents_.{ext}"


def tests_dir_path():
    return os.path.join(".", "pathfinding", "test_cases")


def tests_moving_ai_dir_path():
    return os.path.join("data", "movingai")


def generate_test_suits(tests_config, repeat_for: int):
    os.makedirs(tests_dir_path(), exist_ok=True)
    for map_length, num_agents, density in tests_config:
        env = Environment(
            num_agents=num_agents, map_length=map_length, fix_density=density
        )
        tests = []
        for generated, _ in enumerate(range(repeat_for)):
            tests.append(
                (np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos))
            )
            print(generated)
            env.reset(num_agents=num_agents, map_length=map_length)

        filename = generate_test_filename(map_length, num_agents, density)
        with open(os.path.join(tests_dir_path(), filename), "wb") as file:
            pickle.dump(tests, file)


def generate_test_suits_moving_ai(
    tests_config: list[int], map_filename: str, scenary_filename: str, repeat_for: int
):
    os.makedirs(tests_dir_path(), exist_ok=True)
    map_filename = os.path.join(tests_moving_ai_dir_path(), map_filename)
    scenary_filename = os.path.join(tests_moving_ai_dir_path(), scenary_filename)

    for num_agents in tests_config:
        pkl_tests = []
        tests = get_tests_for_multiple_agents(scenary_filename, num_agents, repeat_for)
        for test_set in tests:
            env = MovingAIBenchmarkingEnvironment(
                num_agents=num_agents,
                map_filename=map_filename,
                test_descriptions=test_set,
            )
            pkl_tests.append(
                (np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos))
            )
        filename = generate_moving_ai_test_filename(
            map_filename,
            scenary_filename,
            num_agents,
        )
        with open(os.path.join(tests_dir_path(), filename), "wb") as file:
            pickle.dump(pkl_tests, file)


def _run_tests(tests_pkl_filename: str, test_generation_fn, singe_test_fn):
    pool = mp.Pool(mp.cpu_count())

    with open(tests_pkl_filename, "rb") as f:
        tests = pickle.load(f)

    tests = test_generation_fn(tests)
    ret = pool.map(singe_test_fn, tests)

    success, avg_step, soft_equality = 0, 0, 0
    for is_successful, ttl_steps, soft_equal in ret:
        success += is_successful
        avg_step += ttl_steps
        soft_equality += soft_equal

    print(f"success rate: {success/len(ret) * 100:.2f}%")
    print(f"soft-success rate: {soft_equality / len(ret) * 100:.2f}%")
    print(f"average step: {avg_step/len(ret)}")
    print()


def test_group(test_group, test_generation_fn, singe_test_fn, is_random=True):
    if is_random:
        length, num_agents, density = test_group
        print(f"test group: {length} length {num_agents} agents {density} density")
        _run_tests(
            os.path.join(
                tests_dir_path(),
                generate_test_filename(length, num_agents, density),
            ),
            test_generation_fn,
            singe_test_fn,
        )
    else:
        num_agents, map_filename, scenary_filename = test_group
        print(
            f"test group: {map_filename} map "
            f"{scenary_filename} scen {num_agents} agents"
        )
        _run_tests(
            os.path.join(
                tests_dir_path(),
                generate_moving_ai_test_filename(
                    map_filename, scenary_filename, num_agents
                ),
            ),
            test_generation_fn,
            singe_test_fn,
        )


def calculate_metrics(env: Environment, steps: int):
    pos_equality = env.agents_pos == env.goals_pos
    soft_equality = (
        pos_equality[:, 0] * pos_equality[:, 1]
    ).sum() / env.agents_pos.shape[0]

    return np.array_equal(env.agents_pos, env.goals_pos), steps, soft_equality


def _dump_to_scen_file(
    scenfile: str,
    tests: list[TestDescription],
    map_h: int,
    map_w: int,
):
    with open(scenfile, "w") as scen:
        print("version 1", file=scen)
        for line_no, test in enumerate(tests):
            test_str = "\t".join(
                map(
                    str,
                    [
                        line_no,
                        test.mapfile,
                        map_h,
                        map_w,
                        test.x0,
                        test.y0,
                        test.x1,
                        test.y1,
                        test.expected_dist,
                    ],
                )
            )
            print(test_str, file=scen)


def generate_scen_for_custom_maps(
    map_filename: str,
    num_agents: int = 8,
    num_tests: int = 10,
):
    path_parts = map_filename.split(os.sep)
    dirpath, mapfile = f"{os.sep}".join(path_parts[:-1]), path_parts[-1]
    scenfile = f"{dirpath}{os.sep}generated_{mapfile.split('.')[-2]}.scen"
    custom_map = read_map(map_filename)

    h, w = custom_map.shape

    pos = np.argwhere(custom_map == 0)

    tests = []
    rng = np.random.default_rng()
    for _ in range(num_tests):
        agents = rng.choice(pos, 2 * num_agents, replace=False)
        start, finish = agents[:num_agents], agents[num_agents:]
        for s, f in zip(start, finish):
            tests.append(
                TestDescription(
                    x0=s[1],
                    y0=s[0],
                    x1=f[1],
                    y1=f[0],
                    expected_dist=-1,
                    mapfile=mapfile,
                )
            )
    _dump_to_scen_file(scenfile, tests, h, w)


if __name__ == "__main__":
    fire.Fire()
