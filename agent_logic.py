import requests
import json
import re

class LocalModelAgent:
    def __init__(self):
        self.memory_buffer = []
        self.max_memory = 10
        self.url = "http://localhost:11434/api/generate"
        self.model_name = "ibm/granite4.1:3b"

    def _format_grid(self, grid):
        """Converts a grid array into a compact string for memory."""
        return " | ".join([" ".join(map(str, row)) for row in grid])

    def update_memory(self, action, status, grid_before, grid_after):
        """Stores the full state transition in memory."""
        before_str = self._format_grid(grid_before)
        after_str = self._format_grid(grid_after)
        
        entry = (
            f"PRE-GRID:  {before_str}\n"
            f"ACTION:    {action} ({status})\n"
            f"POST-GRID: {after_str}"
        )
        
        self.memory_buffer.append(entry)
        if len(self.memory_buffer) > self.max_memory:
            self.memory_buffer.pop(0)


    def get_action_and_thought(self, observation):
        grid_str = "\n".join([" ".join(map(str, row)) for row in observation['grid']])
        memory_str = "\n".join(self.memory_buffer) if self.memory_buffer else "Initial State"

        prompt = f"""<|system|>
You are an autonomous reasoning agent in a 2D grid. 
Try to solve the puzzle.
<|user|>
OBSERVATION (Grid State):
{grid_str}

ACTION HISTORY:
{memory_str}

TASK:
1. try actions and observe the change of environment.
2. make assumption to solve the goal.
3. Select an action: Up, Down, Left, Right, Wait.
Respond ONLY in JSON format: {{"thought": "reasoning here", "action": "Direction"}}
<|assistant|>"""

        try:
            response = requests.post(self.url, json={
                "model": self.model_name, "prompt": prompt, 
                "stream": False, "format": "json",
                "options": {"temperature": 0.0}
            }, timeout=180)
            
            raw_content = response.json().get('response', '{}')
            
            # Robust JSON extraction (removes markdown backticks if present)
            json_str = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_str:
                result = json.loads(json_str.group())
            else:
                result = json.loads(raw_content)

            # Normalize action (e.g., "down" -> "Down")
            action = str(result.get("action", "Wait")).strip().title()
            return {"Thought Log": result.get("thought", ""), "Predicted Action": action}
            
        except Exception as e:
            return {"Thought Log": f"Parsing Error: {e} | Raw: {raw_content[:50]}", "Predicted Action": "Wait"}
