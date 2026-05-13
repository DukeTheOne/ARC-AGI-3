import numpy as np


class ARCAgentEnv:
    """
    A 64x64 grid world environment for ARC-AGI-3.
    """

    def __init__(self, size=64):
        self.size = size
        self.player_pos = [0, 0]
        self.goal_pos = [size - 1, size - 1]
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.player_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self._update_grid()
        return self.observation()

    def _update_grid(self):
        self.grid.fill(0)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        self.grid[self.player_pos[0], self.player_pos[1]] = 1

    def step(self, action):
        """
        Executes an action.
        Actions: 0: Up, 1: Down, 2: Left, 3: Right
        """
        new_r, new_c = self.player_pos

        if action == 0:    # Up
            new_r -= 1
        elif action == 1:  # Down
            new_r += 1
        elif action == 2:  # Left
            new_c -= 1
        elif action == 3:  # Right
            new_c += 1

        # Keep within bounds
        new_r = max(0, min(self.size - 1, new_r))
        new_c = max(0, min(self.size - 1, new_c))

        self.player_pos = [new_r, new_c]

        # Check if solved
        solved = self.player_pos == self.goal_pos

        self._update_grid()

        info = {"status": "Solved" if solved else "InProgress"}
        return self.observation(), solved, info

    def observation(self):
        """Returns the current grid state and player coordinates."""
        return {
            "grid": self.grid.copy(),
            "player_coordinates": tuple(self.player_pos)
        }
