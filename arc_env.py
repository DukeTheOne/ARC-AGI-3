# arc_env.py

import numpy as np

class ARCAgentEnv:
    def __init__(self, size=10):
        self.size = size
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.reset()

    def reset(self):
        self.player_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        return self.observation()

    def observation(self):
        self.grid.fill(0)
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 2
        self.grid[self.player_pos[0], self.player_pos[1]] = 1
        return {"grid": self.grid.copy()}

    def render(self):
        print("\n--- Current Environment State ---")
        for row in self.grid:
            print(" ".join(f"{int(cell):>2}" for cell in row))
        print("---------------------------------")

    # MAKE SURE THIS METHOD IS INCLUDED AND INDENTED CORRECTLY
    def step(self, action):
        old_pos = tuple(self.player_pos)
        r, c = self.player_pos

        if action == 0 and r > 0: r -= 1       # Up
        elif action == 1 and r < self.size-1: r += 1 # Down
        elif action == 2 and c > 0: c -= 1     # Left
        elif action == 3 and c < self.size-1: c += 1 # Right

        self.player_pos = [r, c]
        
        # Check if the player actually moved
        if tuple(self.player_pos) == old_pos and action != 4:
            status = "BLOCKED_BY_BOUNDARY"
        else:
            status = "SUCCESS" if self.player_pos == self.goal_pos else "TRANSITION_COMPLETE"

        return self.observation(), self.player_pos == self.goal_pos, {"status": status}
