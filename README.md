# deep-q-learning-supercabs
Deep Q Learning NN and RL for solving a cab driver 's dilemma to optimize profits

Cab drivers, like most people, are incentivised by a healthy growth in **income**. The goal of this project is to build an RL-based algorithm which can help cab drivers maximise their profits by improving their decision-making process on the field. Following were the issues faced by them:
  * many drivers complain that although their revenues are gradually increasing, their profits are almost flat. 
  * Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to maximise the total profit earned by the driver that day. 
  * If the cab is already in use, then the driver won’t get any requests. 
  * Otherwise, he may get multiple request(s). He can either decide to take any one of these requests or can go ‘offline’, i.e., not accept any request at all.
  * We have to create the MDP environment tha would mimic the env the cab driver interacts with to get his rewards for the action he chooses. The reward structure has to be carefully structured and the step function.
  * Calculation of the reward includes the fare minus the cost of being idle or driving to the pickups point.
  * We choose the second archoitecture, where we passed only the state-values and had the Q-network decide the best action to take.
  * ![image](https://user-images.githubusercontent.com/568941/144803016-2407ff7c-d180-440a-994d-fdd752d41e4f.png)
  * Goals:
  *     - Create the environment
  *     - Build an agent that learns to pick the best request using DQN
  *     - Convergence- We need to converge our results
