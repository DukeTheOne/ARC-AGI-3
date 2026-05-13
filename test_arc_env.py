from arc_env import ARCAgentEnv


def test_initial_state():
    env = ARCAgentEnv()
    obs = env.observation()
    assert obs["player_coordinates"] == (0, 0)
    assert env.grid.shape == (64, 64)


def test_movement():
    env = ARCAgentEnv()

    # Test moving right (Action 3)
    obs, solved, info = env.step(3)
    assert obs["player_coordinates"] == (0, 1)

    # Test moving down (Action 1)
    obs, solved, info = env.step(1)
    assert obs["player_coordinates"] == (1, 1)

    # Test moving left (Action 2)
    obs, solved, info = env.step(2)
    assert obs["player_coordinates"] == (1, 0)

    # Test moving up (Action 0)
    obs, solved, info = env.step(0)
    assert obs["player_coordinates"] == (0, 0)


def test_boundaries():
    env = ARCAgentEnv()

    # Try moving up and left from (0,0)
    env.step(0)
    env.step(2)
    obs = env.observation()
    assert obs["player_coordinates"] == (0, 0)


def test_goal_reached():
    env = ARCAgentEnv(size=3)  # Use small grid for fast testing

    # Initial pos is (0,0), goal is (2,2)
    env.step(3)  # (0,1)
    env.step(3)  # (0,2)
    env.step(1)  # (1,2)
    obs, solved, info = env.step(1)  # (2,2) -> GOAL!

    assert solved is True
    assert info["status"] == "Solved"
