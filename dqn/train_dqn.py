import argparse
import json
import os

import gym
import highway_env
from dqn_agent import DQNAgent

from utils import save_model, set_all_seed

MODELS_DIR = "./models/"
SEED = 36


def load_configuration(config_file):
    with open(config_file, 'r') as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(
        description="Run DQN agent in a highway-env environment.")
    parser.add_argument("-config_file", type=str,
                        help="Path to a configuration file.", default="config/config.json")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(script_dir, args.config_file)
    config = load_configuration(config_file_path)

    # Set up the environment
    env = gym.make(config["env_name"])
    env.reset()
    env.configure(config["env_config"])

    # Seed training
    env.seed(SEED)
    set_all_seed(SEED)

    # Train the agent and save the model
    agent = DQNAgent(env=env, config=config["training_config"])
    agent.train()

    save_model(model=agent.policy_network, dir=MODELS_DIR,
               name=config["training_config"]["output_name"])


if __name__ == "__main__":
    main()
