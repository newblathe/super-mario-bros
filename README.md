# **Super Mario Bros Reinforcement Learning Project**

This project involves training and fine-tuning a reinforcement learning (RL) model to play Super Mario Bros using the `gym-super-mario-bros` environment and reinforcement learning techniques. Follow these sections to set up, preprocess, train, and evaluate the model.

## **Table of Contents**

1. [Installing the Mario Environment](#1-installing-the-mario-environment)
2. [Preprocessing the Environment](#2-preprocessing-the-environment)
3. [Training the RL Model](#3-training-the-rl-model)
4. [Evaluating the Model (1,000,000 Timesteps)](#4-evaluating-the-model-1000000-timesteps)
5. [Fine-Tuning the Best Model (1,000,000 to 2,000,000 Timesteps)](#5-fine-tuning-the-best-model-1000000-to-2000000-timesteps)
6. [Evaluating the Fine-Tuned Model (2,000,000 Timesteps)](#6-evaluating-the-fine-tuned-model-2000000-timesteps)
7. [Advanced Fine-Tuning (2,000,000 to 3,000,000 Timesteps)](#7-advanced-fine-tuning-2000000-to-3000000-timesteps)
8. [Evaluating the Advanced Fine-Tuned Model (3,000,000 Timesteps)](#8-evaluating-the-advanced-fine-tuned-model-3000000-timesteps)
9. [Final Fine-Tuning (3,000,000 to 4,000,000 Timesteps)](#9-final-fine-tuning-3000000-to-4000000-timesteps)
10. [Evaluating the Final Fine-Tuned Model (4,000,000 Timesteps)](#10-evaluating-the-final-fine-tuned-model-4000000-timesteps)
11. [Conclusion](#11-conclusion)
    
## **1. Installing the Mario Environment**

To set up the Super Mario Bros environment:

* **Install the `gym-super-mario-bros` Library**: This library provides the necessary environment for the project.  
* **Import Required Modules**:  
  * Use `JoypadSpace` from `nes_py.wrappers` to simplify controls.  
  * Import `gym_super_mario_bros` and `SIMPLE_MOVEMENT` to configure the environment.  
* **Initialize and Test the Environment**:  
  * Create the environment using `gym_super_mario_bros.make`, apply control simplifications with `JoypadSpace`, and set the render mode.  
  * Run a basic test by taking random actions for 5000 steps to ensure the environment is set up correctly. Use `env.render()` to visualize the agent's actions.

## **2. Preprocessing the Environment**

Prepare the environment for reinforcement learning:

* **Install Required Libraries**:  
  * Install PyTorch and related libraries for building and training models.  
  * Install Stable-Baselines3 and its extra dependencies for reinforcement learning utilities.  
* **Import Necessary Modules**:  
  * Use `GrayScaleObservation` from `gym.wrappers` to convert frames to grayscale.  
  * Import `VecFrameStack` and `DummyVecEnv` from `stable_baselines3.common.vec_env` for vectorization and frame stacking.  
* **Configure the Environment**:  
  * Initialize the environment with `gym_super_mario_bros.make`, apply control simplifications using `JoypadSpace`, and set the render mode.  
  * Apply preprocessing steps:  
    * Convert frames to grayscale.  
    * Wrap the environment in `DummyVecEnv` to manage multiple environments.  
    * Stack frames using `VecFrameStack` to provide the model with a history of frames.

## **3. Training the RL Model**

Train the reinforcement learning model:

* **Setup File Management and Callbacks**:  
  * Import `os` for file management tasks.  
  * Import `PPO` from `stable_baselines3` as the reinforcement learning algorithm.  
  * Import `BaseCallback` from `stable_baselines3.common.callbacks` to define a custom callback for saving models.  
* **Define a Custom Callback**:  
  * Create a `TrainAndLoggingCallback` class inheriting from `BaseCallback` to save the model at specified intervals.  
  * Implement methods to create necessary directories and save the model periodically.  
* **Setup Paths**:  
  * Define paths for saving the models (`checkpoint_dir`) and logs (`log_dir`).  
* **Initialize and Train the Model**:  
  * Create a `PPO` model instance with `CnnPolicy`, specifying the environment, learning rate, number of steps, and log directory.  
  * Train the model using the `model.learn` method, specifying the total number of timesteps and the callback for saving the model.

## **4. Evaluating the Model (1,000,000 Timesteps)**

Evaluate the performance of the trained model:

* **Select and Load the Best Model**:  
  * The model saved at `model_850000` was identified as the best performing in this instance. Modify the line `model = PPO.load('./models/model_850000')` to load any other checkpoint you find suitable based on your evaluation.  
* **Reinitialize the Environment**:  
  * Create the environment using `gym_super_mario_bros.make`.  
  * Apply control simplifications using `JoypadSpace`.  
  * Convert frames to grayscale with `GrayScaleObservation`.  
  * Wrap the environment in `DummyVecEnv` and stack frames with `VecFrameStack`.  
* **Run the Evaluation**:  
  * Start a new episode by resetting the environment.  
  * Use the model to predict actions and apply them to the environment.  
  * Render the environment to visualize the model’s performance.  
* **Close the Environment**:  
  * Close the environment after the evaluation to free up system resources.

## **5. Fine-Tuning the Best Model (1,000,000 to 2,000,000 Timesteps)**

Continue training the best model to improve performance:

* **Setup Environment**:  
  * Reinitialize the environment as described in the previous section.  
* **Setup Directories and Callbacks**:  
  * Define paths for saving the models (`checkpoint_dir`) and logs (`log_dir`).  
  * Set up `TrainAndLoggingCallback` for periodic model saving.  
* **Configure Model**:  
  * Update the model with the new environment.  
  * Set the `tensorboard_log` directory and adjust hyperparameters such as learning rate.  
* **Fine-Tuning Process**:  
  * Continue training the best model (`model_850000`) for an additional 1,000,000 timesteps.

## **6. Evaluating the Fine-Tuned Model (2,000,000 Timesteps)**

Evaluate the fine-tuned model:

* **Select and Load the Fine-Tuned Model**:  
  * Load the model from the saved checkpoint. For example, `model_930000` under the `./models/850000/` directory. Modify the line `model = PPO.load('./models/850000/model_930000')` to load any other checkpoint you find suitable.  
* **Reinitialize the Environment**:  
  * Follow the environment setup steps as described earlier.  
* **Run the Evaluation**:  
  * Start a new episode, use the model to predict actions, and render the environment to observe performance.  
* **Close the Environment**:  
  * Close the environment after evaluation to release resources.

## **7. Advanced Fine-Tuning (2,000,000 to 3,000,000 Timesteps)**

Further refine the model with additional fine-tuning:

* **Setup Environment**:  
  * Reinitialize and preprocess the environment as described in previous sections.  
* **Setup Directories and Callbacks**:  
  * Define paths for saving the models (`checkpoint_dir`) and logs (`log_dir`).  
  * Set up `TrainAndLoggingCallback` for saving the model periodically.  
* **Configure Model**:  
  * Update the model with the new environment.  
  * Set `tensorboard_log` and adjust hyperparameters.  
* **Fine-Tuning Process**:  
  * Continue training the best model (`model_850000`) for another 1,000,000 timesteps.

## **8. Evaluating the Advanced Fine-Tuned Model (3,000,000 Timesteps)**

Assess the performance of the advanced fine-tuned model:

* **Select and Load the Advanced Fine-Tuned Model**:  
  * Load the model from the appropriate checkpoint, such as `model_930000` under `./models/850000/930000/`. Modify the line `model = PPO.load('./models/850000/930000/model_930000')` as needed.  
* **Reinitialize the Environment**:  
  * Follow previous environment setup steps.  
* **Run the Evaluation**:  
  * Start a new episode, predict actions with the model, and render the environment to evaluate performance.  
* **Close the Environment**:  
  * Close the environment after evaluation.

## **9. Final Fine-Tuning (3,000,000 to 4,000,000 Timesteps)**

Further fine-tune the model for optimal performance:

* **Setup Environment**:  
  * Reinitialize the environment with necessary preprocessing steps.  
* **Setup Directories and Callbacks**:  
  * Define paths for saving the models (`checkpoint_dir`) and logs (`log_dir`) and set up `TrainAndLoggingCallback`.  
* **Configure Model**:  
  * Update the model with the new environment and adjust hyperparameters.  
* **Fine-Tuning Process**:  
  * Continue training the best model (`model_850000`) for an additional 1,000,000 timesteps.

## **10. Evaluating the Final Fine-Tuned Model (4,000,000 Timesteps)**

Finally, evaluate the performance of the model:

* **Select and Load the Final Fine-Tuned Model**:  
  * Load the model from the final checkpoint, such as `model_750000` under `./models/850000/930000/930000/`. Modify the line `model = PPO.load('./models/850000/930000/930000/model_750000')` as needed.  
* **Reinitialize the Environment**:  
  * Set up the environment as described before.  
* **Run the Evaluation**:  
  * Start a new episode, use the model for predictions, and render the environment.  
* **Close the Environment**:  
  * Close the environment after evaluation.

## **11. Conclusion**

After training the model for 1,000,000 timesteps, significant improvement was observed as the model began to perform better in the game. Further training up to 2,000,000 and 3,000,000 timesteps did not result in additional substantial improvements. However, after 4,000,000 timesteps, the model demonstrated the capability to clear World 1-1, indicating successful learning and application of gameplay strategies.

(I have uploaded a pre - trained model(model_750000.zip) trained for 4000000 timestemps) 

