import time
from arc_env import ARCAgentEnv
from agent_logic import LocalModelAgent

def run_simulation():
    env = ARCAgentEnv(size=10)
    agent = LocalModelAgent()
    obs = env.reset()
    
    # Action map handles title-cased strings from the agent
    action_map = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3, 'Wait': 4}
    
    for step_num in range(1, 51):
        env.render()
        
        # 1. Capture the grid BEFORE the action
        grid_before = obs['grid'].copy()
        
        # 2. Get Agent decision
        decision = agent.get_action_and_thought(obs)
        action_str = decision["Predicted Action"]
        
        print(f"\n[STEP {step_num}]")
        print(f"AI THOUGHT: {decision['Thought Log']}")
        print(f"AI ACTION : {action_str}")
        
        # 3. Apply action
        action_int = action_map.get(action_str, 4)
        new_obs, solved, info = env.step(action_int)
        
        # 4. Update memory with the transition: Before -> Action -> After
        agent.update_memory(action_str, info["status"], grid_before, new_obs['grid'])
        
        obs = new_obs
        if solved:
            env.render()
            print("\nGOAL REACHED!")
            break
        
        time.sleep(0.6)

if __name__ == "__main__":
    run_simulation()
