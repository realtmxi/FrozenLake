### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
    default="Deterministic-4x4-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary of a nested lists
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""
max_iteration = 100


def policy_evaluation(P, nS, nA, policy, gamma=0.9, theta=1e-3):
    """
    Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
    theta: float
        a small threshold theta > 0 determining accuracy of estimation
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < theta
	Returns
	-------
	V: value_function, np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""
    # initialize the value function V(s)
    V = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0
        for s in range(nS):
            v = 0
            a = policy[s]  # policy action to evaluate
            for prob, next_state, reward, _ in P[s][a]:
                v += prob * (reward + gamma * V[next_state])

            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    ############################
    return V


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(nS):
        action_values = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, _ in P[s][a]:
                action_values[a] += prob * (reward + gamma * value_from_policy[next_state])
        # find the greedy policy with respect to v_{s}
        greedy_action = np.argmax(action_values)
        new_policy[s] = greedy_action
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, theta=1e-3):
    """
    Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	theta: float
		theta parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""
    # Initialization
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    counter = 0
    policy_stable = False
    while (not policy_stable) and counter < max_iteration:
        counter += 1
        # Policy Evaluation
        V = policy_evaluation(P, nS, nA, policy, gamma, theta)

        # Policy Improvement
        new_policy = policy_improvement(P, nS, nA, V, policy, gamma)

        if np.all(policy == new_policy):
            policy_stable = True

        policy = new_policy
    ############################
    return V, policy


def value_iteration(P, nS, nA, gamma=0.9, theta=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	theta: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	V: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            action_values = np.zeros(nA)
            for a in range(nA):
                state_value = 0
                for prob, next_state, reward, _ in P[s][a]:
                    state_value += prob * (reward + gamma * V[next_state])
                action_values[a] = state_value
            best_action = np.argmax(action_values)
            V[s] = action_values[best_action]
            policy[s] = best_action
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    ############################
    return V, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    observation, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        action = policy[observation]
        observation, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done and reward == 1:
            print("You reach the goal in {} steps.".format(t))
            break
        elif done and reward == 0:
            print("you fell in a hole!")
    env.render()
    if not done:
        print(
            "The agent didn't reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode, map_name="4x4", is_slippery=False)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, theta=1e-3)
    render_single(env, p_pi, 100)

    # print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    #
    # V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, theta=1e-3)
    # render_single(env, p_vi, 100)
