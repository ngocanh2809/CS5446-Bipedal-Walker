This sub-folders contain files corresponding to each variation of hyperparameters tested. 

To reproduce results, go to each sub-folder, in BipedalWalker-v3_PPO.py, uncomment code under agent.test(20) and run the code.

if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    env_name = 'BipedalWalker-v3'
    agent = PPOAgent(env_name)
    # agent.run_batch() # train as PPO
    # agent.run_multiprocesses(num_worker = 12)  # train PPO multiprocessed (fastest)
    agent.test(20)
    # agent.visualize_iteration(1)