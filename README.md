# FrozenLake
This environment is part of the Toy Text environments which contains general information about the environment.
https://gymnasium.farama.org/environments/toy_text/frozen_lake
[](https://gymnasium.farama.org/_images/frozen_lake.gif)
## Action Space
- 0: Move left
- 1: Move down
- 2: Move right
- 3: Move up

## Rewards
Reward schedule:
- Reach goal: +1
- Reach hole: 0
- Reach frozen: 0

## Episode End
The episode ends if the following happens:
- The player moves into a hole
- The player reaches the goal at max(nrow) * max(ncol) - 1