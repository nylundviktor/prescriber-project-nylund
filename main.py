import gymnasium as gym
import optuna
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_learning_curve(log_path, window_size=100):
    """Generates and saves a plot of the learning curve.
    
    Currently hardcodes the output filename (learning_curve.png) — 
    could be made more flexible to avoid overwriting or for multiple experiments.

    Depends on Stable Baselines/Gym monitor log format — if you switch logging style, 
    you may need minor adjustments.

    Could optionally return the figure object if you want to further manipulate or display it inline.
    """
    try:
        log_data = pd.read_csv(log_path + ".monitor.csv", skiprows=1)
        cumulative_timesteps = log_data['l'].cumsum()
        moving_avg = log_data['r'].rolling(window=window_size).mean()

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_timesteps, log_data['r'], alpha=0.3, label='Per-Episode Reward')
        plt.plot(cumulative_timesteps, moving_avg, color='red', linewidth=2, label=f'Moving Average (window={window_size})')
        plt.title("Learning Curve of the Final Agent")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("learning_curve.png")
        plt.close()
        print("\nLearning curve plot saved to learning_curve.png")
    except FileNotFoundError:
        print("\nCould not find monitor log file. Skipping learning curve plot.")


def plot_optuna_study(study):
    """Generates and saves a plot of the Optuna study.
    

    Currently hardcodes the output filename (optuna_study.png) — could be made flexible for multiple studies.

    Only plots trial value vs. trial number; for deeper analysis, 
    you might also want parameter importance plots (optional).

    Could optionally return the figure object for further manipulation in scripts or notebooks.
    """
    if not study.trials:
        print("No trials found in the study to plot.")
        return
    trial_values = [t.value for t in study.trials if t.value is not None]
    if not trial_values:
        print("No successful trials with values to plot.")
        return
    trial_numbers = [t.number for t in study.trials if t.value is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(trial_numbers, trial_values, marker='o', linestyle='--')
    if study.best_trial and study.best_trial.value is not None:
        best_trial_num = study.best_trial.number
        best_trial_val = study.best_trial.value
        plt.scatter(best_trial_num, best_trial_val, s=120, c='red', zorder=5, label=f'Best Trial (#{best_trial_num})')
    plt.title("Optuna Hyperparameter Optimization History")
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("optuna_study.png")
    print("Optuna study plot saved to optuna_study.png")
    plt.close()


# Hyperparameter tuning with Optuna
def objective(trial):
    """The objective function for Optuna to maximize.
    
    
    Currently creates and wraps a new environment for each trial, 
    which is fine for small-scale searches, but can be slow for large Optuna searches. 
    Could be improved later with vectorized or pre-wrapped envs if needed.

    total_timesteps=10000 is very small; for real searches you'd increase it, 
    but this is just structural feedback.

    Hardcoded policy type "MlpPolicy" — works, but if you switch to CNN for image input, 
    you'd need to generalize.
    """
    # Only for training
    trial_env = Monitor(WarpFrame(gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False), 
        width=84, height=84))

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True) 
    gamma = trial.suggest_float("gamma", 0.99, 0.9999) 
    layer_size = trial.suggest_categorical("layer_size", [64, 128, 256]) 
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) 
    tau = trial.suggest_float("tau", 0.001, 0.1) 
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.5) 
    # Number of transitions stored for experience replay
    buffer_size = trial.suggest_categorical("buffer_size", [20_000, 50_000, 100_000]) 
    # Agent explores early on
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3) 
    # How often the network is updated
    target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 5000]) 

    # How often the policy is updated (every n environment steps)
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8]) 
    # Number of gradient updates per train step
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4]) 

    net_arch = [layer_size, layer_size]
    policy_kwargs = dict(net_arch=net_arch)
    
    model = DQN(
        "MlpPolicy", 
        trial_env, 
        learning_rate=learning_rate, 
        gamma=gamma,
        batch_size=batch_size,
        tau=tau,
        exploration_final_eps=exploration_final_eps,
        buffer_size=buffer_size,
        exploration_fraction=exploration_fraction,
        target_update_interval=target_update_interval,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    # 5k-10k OK
    model.learn(total_timesteps=5000)

    # fresh evaluation environment (callable)
    eval_env_class = lambda: Monitor( WarpFrame( gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )))
    
    # Evaluate using the separate environment. 10-20
    mean_reward, _ = evaluate_model(model, env_class=eval_env_class, n_eval_episodes=10, use_predict=True)
    print(f"Trial {trial.number} completed and values stored. \nMean reward: {mean_reward:.2f}.")
    trial_env.close()
    return mean_reward


def train_final_model(best_trial, total_timesteps=50_000, log_dir="logs/"):
    """
    Train the final DQN model using the best hyperparameters from Optuna.
    """
    # log path
    final_log_path = os.path.join(log_dir, "final_model_logs")
    os.makedirs(final_log_path, exist_ok=True)

    # find best hyperparameters
    best_params = best_trial.params.copy()
    final_layer_size = best_params.pop("layer_size")
    final_policy_kwargs = dict(net_arch=[final_layer_size, final_layer_size])

    # Create and wrap environment with Monitor for logging
    final_env = gym.make("CarRacing-v3", continuous=False)
    final_env = Monitor(final_env, final_log_path)

    # Create DQN model with best hyperparameters
    final_model = DQN(
        "MlpPolicy",
        final_env,
        policy_kwargs=final_policy_kwargs,
        **best_params,
        verbose=1)

    # Train model 20k test, 200k ok training, 500k+ strong training
    final_model.learn(total_timesteps=total_timesteps)

    # Save trained model
    final_model.save(os.path.join(log_dir, "best_carracing_model"))
    print(f"Final model saved to {os.path.join(log_dir, ' zip')}")

    return final_model, final_log_path


def evaluate_model(model, env_class, n_eval_episodes=100, use_predict=True):
    """
    Evaluates a given RL model on the provided environment class.

    
    The function creates a new environment instance, which is fine, but you might want it to accept a 
    pre-created environment in some workflows (e.g., vectorized or wrapped environments).

    Could eventually support vectorized evaluation for efficiency if you have multiple environments.

    The loop uses done — in Gymnasium v3 you might also need to handle terminated and truncated separately, 
    but this is a minor detail for the overarching structure.
    """

    print("\n--- Evaluating Optuna Trial Model Performance ---")
    
    eval_env = env_class()
    rewards = []
    for ep in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        ep_reward = 0.0
        
        while not done:
            if use_predict:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = model(obs)  # Monte Carlo policy
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        
        rewards.append(ep_reward)
    
    eval_env.close()
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    print(f"Optuna Trial Model: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


if __name__ == "__main__":
    # Check and create logs
    LOG_DIR = "logs/"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Runtime variables
    STORAGE_PATH = "sqlite:///my_study.db" # database
    STUDY_NAME = "car_racing_dqn"
    NUM_TRIALS_TO_RUN = 30 # Number of trial runs

    # Create/load Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True
    )

    # Run Optuna optimization
    # Avoid additional trials | Increase variable for more runs
    if len(study.trials) < NUM_TRIALS_TO_RUN:
        study.optimize(objective, n_trials=NUM_TRIALS_TO_RUN - len(study.trials))
    else:
        print(f"Study already has {len(study.trials)} trials. Skipping optimization.")

    print("\n--- Best Trial Information ---")
    best_trial = study.best_trial
    if best_trial:
        print(f"  Value (Mean Reward): {best_trial.value:.2f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
            
        # Train the final model
        final_model, final_log_path = train_final_model(best_trial, 10_000, log_dir=LOG_DIR)
        # Evaluate
        mean_reward, std_reward = evaluate_model(final_model, n_eval_episodes=30)
        
        # Images & Plotting
        plot_optuna_study(study)
        plot_learning_curve(final_log_path)

    else:
        print("No successful trials were completed. Cannot train or evaluate a final model.")


'''
run_optuna_search()
evaluate_and_plot(model, log_path)

gymnasium           1.2.1 
stable_baselines3   2.3.2 
torch               2.5.1
'''