def test_import_dhc():
    from pathfinding.models.dhc import DHCNetwork  # noqa


def test_import_env():
    from pathfinding.environment import Environment  # noqa


def test_import_buffer():
    from pathfinding.models.dhc import LocalBuffer  # noqa


def test_import_worker():
    from pathfinding.models.dhc import Actor, Learner, GlobalBuffer  # noqa
