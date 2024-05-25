## Robot navigation in crowd
- This project focus on make a simple baseline, minimal configuration, easy to train and test and simulation based on recent researches on crowd-aware robot navigation
- In this project, I use Q-learning with 80 actions (5 level of magnitude and 16 orientation),using (RVO2)[https://github.com/sybrenstuvel/Python-RVO2/] for generate human interaction based on ORCA algorithms. I use LSTM [3] to encode crowd information.
#### Result
*Training*
![alt text](https://github.com/phuongboi/robot-navigation-in-crowd/blob/main/figures/fig_660000.png)
*Testing* (updating)

#### How to use
- Training: `python train.py`
- Testing: `python test.py`(current test result don't meet the expectation)
- Simulation in Coppelisim: (updating)

#### Reference
* [1] https://github.com/vita-epfl/CrowdNav
* [2] Chen, Yu Fan, et al. "Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning." 2017 IEEE international conference on robotics and automation (ICRA). IEEE, 2017.
* [3] Everett, Michael, Yu Fan Chen, and Jonathan P. How. "Motion planning among dynamic, decision-making agents with deep reinforcement learning." 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018.
* [4] Chen, Changan, et al. "Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning." 2019 international conference on robotics and automation (ICRA). IEEE, 2019.
