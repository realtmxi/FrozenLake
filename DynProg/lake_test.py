import gymnasium as gym
from gymnasium.envs.registration import register

env = gym.make("FrozenLake-v1", render_mode="human", is_slippery="False")

# test
episode_reward = 0
done = False

for t in range(5):
    print("The agent begins episode {}.".format(t))
    observation, info = env.reset()
    while not done:
        # Render the current state
        env.render()

        # Prompt the user for the action (0: Left, 1: Down, 2: Right, 3: Up)
        action = int(input("Enter the action (0: Left, 1: Down, 2: Right, 3: Up): "))

        # Perform the action in the environment
        observation, reward, done, truncated, _ = env.step(action)
        episode_reward += reward

        print("Agent took action: {}".format(action))
        print("Received reward: {}".format(reward))
        print("Episode reward: {}".format(episode_reward))
        print()

        # Check if the episode is done
        if done:
            if reward == 1:
                print("Congratulations! The agent reached the goal.")
            else:
                print("Oops! The agent fell into a hole.")
        if done or truncated:
            observation, info = env.reset()
        done = False
env.close()



