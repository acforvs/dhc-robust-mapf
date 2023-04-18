import fire
from collections import defaultdict
import numpy as np
import os
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

    while not done and env.steps < GENERAL_CONFIG["max_episode_length"]:
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
    model_number="60000",
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

        def func(x):
            return _test_generation_fn_random(x, network)

    else:

        def func(x):
            return _test_generation_fn_moving_ai(x, network)

    for group in test_groups:
        yield test_group(group, func, _test_one_case, is_random=is_random_maps), group


def latex_communication_table(
    model_communication_id: str = "337500",
    model_nocommunication_id: str = "310000_nocomm",
    density=0.1,
    agents=(4, 8, 16, 32),
    maps=(10, 20, 30, 40),
):
    test_groups = []
    for num_agents in agents:
        for map_size in maps:
            if density == 0.3 and map_size == 10 and num_agents == 32:
                continue
            test_groups.append((map_size, num_agents, density))
    header = f"""\\newpage
\\begin{{longtable}}[htb!]{{cc|ccc}}
\caption{{TODO CAPTION}} \label{{table:TODO-TABLE-LABEL}}\\\\

\\toprule
\multicolumn{{2}}{{c}}{{Map configuration}} & \multirow{{2}}{{*}}{{Metrics}} & \multicolumn{{2}}{{c}}{{Density {density}}}  \\\\
\# Agents                    & Size         &                                   & DHC (ours)       & DHC (original training) \\\\
\midrule
\endfirsthead

\multicolumn{{5}}{{c}}%
{{{{Table \\thetable\ continued from previous page}}}} \\\\
\\toprule
\multicolumn{{2}}{{c}}{{Map configuration}} & \multirow{{2}}{{*}}{{Metrics}} & \multicolumn{{2}}{{c}}{{Density {density}}}  \\\\
\# Agents                    & Size         &                                   & DHC (ours)       & DHC (original training) \\\\
\midrule
\endhead

\midrule
\multicolumn{{5}}{{r}}{{Continued on next page}} \\
\endfoot

\\bottomrule
\endlastfoot"""

    comm = defaultdict(lambda: defaultdict(list))
    nocomm = defaultdict(lambda: defaultdict(list))
    for res, (size, num_agents, density) in test_model(
        test_groups, model_nocommunication_id
    ):
        nocomm[num_agents][size] = res
    for res, (size, num_agents, density) in test_model(
        test_groups, model_communication_id
    ):
        comm[num_agents][size] = res

    num_maps = len(maps)
    table = []
    metrics = ["CSR, \%", "ISR, \%", "Makespan"]
    for i, num_agents in enumerate(agents):
        sector = f"\multirow{{{num_maps * 3}}}{{*}}{{{num_agents}}} "
        for map_id, map_size in enumerate(maps):
            row = ["", f" \multirow{{3}}{{*}}{{${map_size} \\times {map_size}$}} "]
            nocomm_list = nocomm[num_agents][map_size]
            comm_list = comm[num_agents][map_size]
            for metric_id, ((a_mean, a_std), (b_mean, b_std)) in enumerate(
                zip(nocomm_list, comm_list)
            ):
                if metric_id != 0:
                    row.append("")
                row.append(metrics[metric_id])
                if (metric_id != 2 and a_mean >= b_mean) or (
                    metric_id == 2 and a_mean <= b_mean
                ):  # makespan: the less, the better
                    if metric_id != 0:  # std for CSR doesn't make sense
                        row.append(
                            f"\\textbf{{{a_mean:.2f}}} $\pm$ \\textbf{{{a_std:.2f}}}"
                        )
                        row.append(f"${b_mean:.2f} \pm {b_std:.2f}$ \\\\ \n")
                    else:
                        row.append(f"\\textbf{{{a_mean:.2f}}}")
                        row.append(f"{b_mean:.2f} \\\\ \n")
                else:
                    if metric_id != 0:  # std for CSR doesn't make sense
                        row.append(f"${a_mean:.2f} \pm {a_std:.2f}$")
                        row.append(
                            f"\\textbf{{{b_mean:.2f}}} $\pm$ \\textbf{{{b_std:.2f}}} \\\\ \n"
                        )
                    else:
                        row.append(f"{a_mean:.2f}")
                        row.append(f"\\textbf{{{b_mean:.2f}}} \\\\ \n")
            sector += " & ".join(row)
            if map_id != num_maps - 1:
                sector += "\cline{2 - 5} \n"
            else:
                sector += "\n"
        if i == len(agents) - 1:
            sector += "\\bottomrule\n"
        else:
            sector += "\midrule\n"
        table.append(sector)
    footer = "\\end{longtable}\n"
    print(header + "\n".join(table) + footer)


if __name__ == "__main__":
    fire.Fire()
