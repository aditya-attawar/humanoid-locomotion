import gymnasium as gym
import imageio
from tqdm import tqdm
# import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import nn  # Import activation functions
import multiprocessing
# import os
import cProfile
import pstats

# print(gym.envs.registry.keys())
train_model = True
test_model = True
save_video = True

env_id = "HumanoidStandup-v5"  # The environment ID
save_path = "ppo_humanoid_model"  # Path to save the model

if train_model:

    # Define customizable policy structure
    print("Defining policy structure")
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network architecture: 2 layers of 256 units each
            vf=[256, 256],  # Value network architecture: 2 layers of 256 units each
        ),
        activation_fn=nn.ReLU,  # Use the actual ReLU function, not a string
    )

    # Training Parameters
    print("Defining training parameters...")
    total_timesteps = 5_000_000  # Total training steps
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
    cProfile.run("model.learn(total_timesteps=10000)", profiler_output)

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


 # Test the trained model
if test_model:
    if not train_model:
        # Load the model
        print("Loading the model...")
        model = PPO.load(save_path)
        print(f"Model loaded from {save_path}")

    # env = gym.make("HumanoidStandup-v5", render_mode=("rgb_array" if save_video else "human"), width=1280, height=720)
    # print(env.action_space)

    if save_video:
        frames = []

    print("Testing the trained model...")
    test_env = gym.make(env_id, render_mode=("rgb_array" if save_video else "human"), width=1280, height=720)
    observation, _ = test_env.reset(seed=42)
    # print(observation.shape[0])

    max_steps = 1000
    for _ in tqdm(range(max_steps), total=max_steps, unit="step", desc="Progress"):
        action, _ = model.predict(observation, deterministic=True)
        # action = test_env.action_space.sample()
        observation, reward, terminated, truncated, _ = test_env.step(action)
        if save_video:
            frames.append(test_env.render())  # Collect frames
        # print(reward)
        # print(observation, reward, terminated, truncated)
        if terminated or truncated:
            # print(f"Episode ended at step {step}")
            observation, _ = test_env.reset()

    test_env.close()

    # Save the frames as a video
    if save_video:
        print("Saving the video...")        
        with imageio.get_writer("humanoid_standup.mp4", fps=30) as video:
            for frame in frames:
                video.append_data(frame)
        print("Video saved.")
