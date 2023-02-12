import numpy as np


import fire
from dataclasses import dataclass


@dataclass
class TestDescription:
    x0: int
    y0: int
    x1: int
    y1: int
    expected_dist: float
    mapfile: str | None


def _transform(line: str) -> list[int]:
    return [
        int(ch) for ch in line.replace("@", "1").replace(".", "0").replace("T", "1")
    ]


def read_map(mapfile: str) -> np.array:
    with open(mapfile, "r") as map_file:
        map_file.readline()
        _, h = map_file.readline().split(" ")
        _, w = map_file.readline().split(" ")
        h, w = int(h), int(w)
        map_file.readline()
        lines = map_file.read().split("\n")
        map = [_transform(line) for line in lines if line]

        h_read, w_read = len(map), len(map[0])
        if h_read != h or w_read != w:
            raise ValueError(
                "Size of the map read is not equal to the expected size from MovingAI,"
                f"({h_read}, {w_read}) != ({h}, {w})"
            )

        return np.array(map)


def get_map_density(mapfile: str) -> float:
    map = read_map(mapfile)
    return (map == 1).sum() / (map.shape[0] * map.shape[1])


def read_scenario_from_file(scenfile: str):
    tests = []

    with open(scenfile, "r") as scen_file:
        scen_file.readline()
        for line in scen_file:
            test_no, map_no, h, w, y0, x0, y1, x1, exp_dst = line.split("\t")
            x0, y0, x1, y1, exp_dst = int(x0), int(y0), int(x1), int(y1), float(exp_dst)
            tests.append(TestDescription(x0, y0, x1, y1, exp_dst, map_no))
    return tests


def get_tests_for_multiple_agents(
    scenfile: str = None, num_agents: int = 5, max_num_tests: int = 200
) -> list[list[TestDescription]]:
    scens = read_scenario_from_file(scenfile)
    tests = []
    num_tests = min(max_num_tests * num_agents - 1, len(scens) - num_agents + 1)
    for i in range(0, num_tests, num_agents):
        tests.append(scens[i : i + num_agents])
    return tests


if __name__ == "__main__":
    fire.Fire()

# success rate: 0.00%
# soft-success rate: 45.78%
# average step: 256.0

# success rate: 0.00%
# soft-success rate: 85.47%
# average step: 512.0
#
# success rate: 70.00%
# soft-success rate: 95.78%
# average step: 773.1
