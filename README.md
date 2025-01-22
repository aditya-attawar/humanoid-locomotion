
# Humanoid Locomotion Simulation with PPO

This project focuses on training a reinforcement learning (RL) agent to make a humanoid stand up and perform locomotion tasks using the `HumanoidStandup-v5` environment from Mujoco. The agent is trained using the Proximal Policy Optimization (PPO) algorithm provided by the Stable-Baselines3 library.

## **Project Overview**
The primary goal of this project is to:
- Train a humanoid to stand up and potentially begin locomotion.
- Leverage reinforcement learning techniques to optimize the agent's performance.
- Utilize parallel environments and efficient system resource management for faster training.

---

## **Progress**
### **Current Achievements:**
1. **Environment Setup:**
   - Created a Mujoco-based `HumanoidStandup-v5` environment for training and testing.

2. **Training Pipeline:**
   - Successfully utilized the PPO algorithm with customizable neural network architectures.
   - Trained the model for various timesteps using vectorized environments (`make_vec_env`) to leverage parallel processing.
   - Optimized training by dynamically adjusting the number of environments (`n_envs`) and leveraging all available CPU cores.

3. **Testing Pipeline:**
   - Implemented a test loop to visualize and evaluate the trained model in real-time using `render_mode="human"`.

4. **Performance Optimizations:**
   - Monitored CPU usage and process prioritization using `htop` and `nice`.
   - Experimented with `n_steps`, `batch_size`, and `n_envs` to maximize training speed.

5. **Profiling:**
   - Added Python `cProfile` support to identify bottlenecks in the training pipeline.

---

## Requirements
Ensure you have the following installed:
- Python 3.8+
- Mujoco and Mujoco-py
- Stable-Baselines3
- Gymnasium

<!-- Install dependencies with:
```bash
pip install -r requirements.txt
``` -->

---

## Usage

### **1. Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
```

### **2. Train the Agent**
```bash
python test.py
```
- The training script will train the PPO model for the specified timesteps and save the model as `ppo_humanoid_model.zip`.

### **3. Test the Trained Model**
```bash
python test.py  # Testing is included in the script after training
```
The humanoid's behavior will be rendered in real-time.

### **4. Monitor System Performance**
- Use `htop` (Linux/Mac) or Task Manager (Windows) to monitor CPU utilization during training.
- Adjust `n_envs` and `n_steps` for optimal performance.

---

## **Optimization Tips**
- **CPU Usage:** Use `multiprocessing.cpu_count()` to dynamically set `n_envs` based on available cores.
- **Priority:** Run the script with higher priority:
  ```bash
  sudo nice -n -10 python train_humanoid.py
  ```
- **Profiling:** Use `cProfile` to identify bottlenecks:
  ```bash
  cProfile.run("model.learn(total_timesteps=10000)", "training_profile.prof")
  ```

---

## **To-Do**
- [ ] Implement reward shaping for better control of standing behavior.
- [ ] Experiment with alternate RL algorithms (e.g., SAC, TRPO).
- [ ] Visualize training progress with TensorBoard.
- [ ] Save videos of testing sessions for documentation.

---

## **References**
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Mujoco Physics Engine](https://mujoco.org/)
- [Gymnasium Environments](https://farama.org/Gymnasium/)

---

<!-- ## **License**
This project is licensed under the MIT License. -->
