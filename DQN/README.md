# Deep Q-Network

Need to implement:
- [ ] `ReplayBuffer` class with the following functionality:
    - Sample n items from the buffer uniformly.
    - Be able to set a max size `capacity` that flushes out the oldest data if overflowing.
- [ ] `DQN` class with the following parameters:
    - `K`: the number of gradient steps per data collection cycle
    - `N`: the number of data collections per target network parameter 
           update (the frequency of saving phi' <- phi)
    - `K=1, N=1` is the implementation for the classic DQN algorithm. Try playing around with this.
 - [ ] Double Q-learning
      - Using the current network phi to evaluate the action and the saved network phi' to compute the 
        predicted value at that state/action.
 - [ ] Plot experiments with sample environments (of actual return/predicted Q-values vs. training iterations)
       with three different settings:
    1. DQN with replay buffer w/o target network
    2. Classic DQN with replay buffer, target network
    3. Double Q-learning implementation