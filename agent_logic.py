import random

class LocalModelAgent:
    def __init__(self):
        # Keeps track of the last 5 moves and their results
        # e.g., 'Moved Right -> Hit Wall'
        self.memory_buffer = []
        self.max_memory = 5
        self.possible_actions = ['Up', 'Down', 'Left', 'Right', 'Wait']

    def update_memory(self, action, result):
        """
        Updates the memory buffer with the latest action and its result.
        """
        memory_entry = f"{action} -> {result}"
        self.memory_buffer.append(memory_entry)
        # Keep only the last 5 entries
        if len(self.memory_buffer) > self.max_memory:
            self.memory_buffer.pop(0)

    def get_action_and_thought(self, observation):
        """
        Takes the grid observation and outputs a Thought Log and a Predicted Action.
        Structured to easily plug in a local Llama-3.2-1B model later.
        """
        # Format the memory for the prompt
        memory_str = "\n".join(self.memory_buffer) if self.memory_buffer else "None"
        
        # ---------------------------------------------------------
        # TODO: Future Llama-3.2-1B integration point
        # prompt = f'''
        # System: You are an agent navigating a grid. Here is your recent memory:
        # {memory_str}
        # 
        # Current Observation:
        # {observation}
        # 
        # Output your Thought Log explaining what you see, and then output your Predicted Action.
        # '''
        # response = llama_model.generate(prompt)
        # thought_log, action = parse_response(response)
        # ---------------------------------------------------------

        # Dummy random logic for now
        thought_log = f"Observation: {observation}. Memory: [{memory_str}]. Based on this, I will take a random action."
        action = random.choice(self.possible_actions)

        return {
            "Thought Log": thought_log,
            "Predicted Action": action
        }
