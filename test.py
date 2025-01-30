import gymnasium as gym
import imageio
from tqdm import tqdm
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from torch import nn  # Import activation functions
import multiprocessing
import os
import cProfile
import pstats
from huggingface_sb3 import load_from_hub
import shutil
from gymnasium.spaces import Box
from pathlib import Path


class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to match model's requirement
        self.observation_space = Box(-np.inf, np.inf, (45,), dtype=np.float32)

    def observation(self, obs):
        # Preprocess observation to match the model's expected shape
        # Here we assume the relevant data is in the first 45 dimensions
        # Adjust this if the model expects different preprocessing
        return obs[:45]


# Wrap the environment
env_id = "HumanoidStandup-v5"
test_env = gym.make(env_id)
test_env = PreprocessObservationWrapper(test_env)
observation, _ = test_env.reset()
print("Processed observation shape:", observation.shape)
assert observation.shape == (45,), "Observation shape mismatch"


# print(gym.envs.registry.keys())
train_model = False
test_model = True
save_video = True

env_id = "HumanoidStandup-v5"  # The environment ID
save_path = "humanoid_model"  # Path to save the model

if train_model:

    # Define customizable policy structure
    print("Defining policy structure")
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512],  # Policy network architecture: 2 layers of 256 units each
            vf=[512, 512],  # Value network architecture: 2 layers of 256 units each
        ),
        activation_fn=nn.ReLU,  # Use the actual ReLU function, not a string
    )

    # Training Parameters
    print("Defining training parameters...")
    total_timesteps = 10_000_000  # Total training steps
    # checkpoint_interval = 100_000  # Save the model every 100k steps

    # Create the environment
    print("Creating the environment...")
    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f"Detected {num_cores} CPU cores.")
    env = make_vec_env(env_id, n_envs=num_cores, env_kwargs={"render_mode": "none"})  # Use vectorized environments for efficient training

    # Initialize the PPO model
    print("Initializing the PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,  # Number of steps per environment before update
        batch_size=128,  # Batch size for updates
        n_epochs=10,  # Number of optimization epochs per update
        verbose=1
    )

    # Profiling the training
    profiler_output = "training_profile.prof"
    print("Profiling training...")
    cProfile.run("model.learn(total_timesteps=10000000)", profiler_output)

    # Analyzing the profiling output
    print("Analyzing profiling results...")
    with open("profiling_results.txt", "w") as f:
        stats = pstats.Stats(profiler_output, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")  # Sort by cumulative time
        stats.print_stats()
    print("Profiling completed. Results saved to profiling_results.txt")

    # # Train the model
    # print("Training started...")
    # model.learn(total_timesteps=total_timesteps)
    # print("Training finished.")

    # Save the model
    print("Saving the model...")
    model.save(save_path)
    print(f"Model saved to {save_path}")

else:
    from huggingface_sb3 import load_from_hub
    print("Downloading the model...")
    filename="humanoidstandup-v5-sac-expert.zip"
    model_path_alias = load_from_hub(
        repo_id="farama-minari/HumanoidStandup-v5-SAC-expert",
        filename=filename,
    )
    model_path = os.path.realpath(model_path_alias)
    temp_filename = Path(model_path).name
    print(f"Model downloaded to {model_path}")
    print("Moving the model to parent directory...")
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    shutil.move(model_path, os.getcwd())
    os.rename(temp_filename, save_path+".zip")
    print(f"Model moved to ", "./"+save_path+".zip")


# Test the trained model
if test_model:
    if not train_model:
        # Load the model
        print("Loading the model...")
        model = SAC.load(save_path)
        print(f"Model loaded from {save_path}")

    if save_video:
        # Define the output video writer
        video_writer = imageio.get_writer("humanoid_standup.mp4", fps=30)

    print("Testing the trained model...")
    test_env = gym.make(env_id, render_mode=("rgb_array" if save_video else "human"), width=1280, height=720)
    test_env = PreprocessObservationWrapper(test_env)  # Apply wrapper here
    observation, _ = test_env.reset(seed=42)

    print("Environment observation space:", test_env.observation_space)
    print("Expected observation space from the model:", model.observation_space)

    max_steps = 1000
    for _ in tqdm(range(max_steps), total=max_steps, unit="step", desc="Progress"):
        action, _ = model.predict(observation, deterministic=True)
        # action = test_env.action_space.sample()
        observation, reward, terminated, truncated, _ = test_env.step(action)
        if save_video:
            video_writer.append_data(test_env.render()) # append the frame to the video
        # print(reward)
        # print(observation, reward, terminated, truncated)
        if terminated or truncated:
            # print(f"Episode ended at step {step}")
            observation, _ = test_env.reset()

    video_writer.close()
    test_env.close()
    print("Video saved.")
