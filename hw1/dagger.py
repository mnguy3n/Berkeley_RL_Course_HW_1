#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python3 dagger.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib

HYPER_PARAMETER_DICT = {
  'Ant': {
    'batch_size': 200,
    'learning_rate': 0.01,
    'num_relus': 30,
    'beta_A': 0.0005,
    'beta_B': 0.0005,
    'beta_output': 0.0005,
    'num_steps': 50001
  },
  'HalfCheetah': {
    'batch_size': 200,
    'learning_rate': 0.005,
    'num_relus': 30,
    'beta_A': 0.0005,
    'beta_B': 0.0005,
    'beta_output': 0.0005,
    'num_steps': 50001
  },
  'Hopper': {
    'batch_size': 200,
    'learning_rate': 0.01,
    'num_relus': 30,
    'beta_A': 0.00001,
    'beta_B': 0.00001,
    'beta_output': 0.0001,
    'num_steps': 50001
  },
  'Humanoid': {
    'batch_size': 200,
    'learning_rate': 0.01,
    'num_relus': 30,
    'beta_A': 0.0005,
    'beta_B': 0.0005,
    'beta_output': 0.0005,
    'num_steps': 50001
  },
  'Reacher': {
    'batch_size': 200,
    'learning_rate': 0.001,
    'num_relus': 30,
    'beta_A': 0.001,
    'beta_B': 0.001,
    'beta_output': 0.001,
    'num_steps': 50001
  },
  'Walkder2d': {
    'batch_size': 200,
    'learning_rate': 0.001,
    'num_relus': 30,
    'beta_A': 0.001,
    'beta_B': 0.001,
    'beta_output': 0.001,
    'num_steps': 50001 
  }
}

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def get_initial_data(env, policy, max_steps, args):
  returns = []
  observations = []
  actions = []
  for i in range(args.num_rollouts):
    if args.print_debug:
      print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done:
      action = policy.act(obs)
      observations.append(obs)
      actions.append(action)
      obs, r, done, _ = env.step(action)
      totalr += r
      steps += 1
      if args.render:
        env.render()
      if args.print_debug and steps % 100 == 0:
        print("%i/%i"%(steps, max_steps))
      if steps >= max_steps:
        break
    returns.append(totalr)
  print('policy returns', returns)
  print('policy mean return', np.mean(returns))
  print('policy std of return', np.std(returns))
  return {'observations': np.array(observations), 'actions': np.array(actions)}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('expert_policy_file', type=str)
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--max_timesteps', type=int)
  parser.add_argument('--num_rollouts', type=int, default=20,
                      help='Number of expert roll outs')
  parser.add_argument('--print_debug', type=bool, default=False)
  args = parser.parse_args()

  print('Loading expert policy')
  module_name = args.expert_policy_file.replace('/', '.')
  if module_name.endswith('.py'):
    module_name = module_name[:-3]
  policy_module = importlib.import_module(module_name)

  env, policy = policy_module.get_env_and_policy()
  max_steps = args.max_timesteps or env.spec.timestep_limit
  expert_data = get_initial_data(env, policy, max_steps, args)
    
  ## behavioral cloning ##
  total_data_size = len(expert_data['observations'])
  training_size = int(total_data_size * 0.7)
  validation_size = int(total_data_size * 0.2)
  testing_size = int(total_data_size * 0.1)

  # separating data
  training_x = expert_data['observations'][0:training_size]
  training_y = expert_data['actions'][0:training_size]
  validation_x = expert_data['observations'][training_size:training_size + validation_size]
  validation_y = expert_data['actions'][training_size:training_size + validation_size]
  testing_x = expert_data['observations'][training_size + validation_size:]
  testing_y = expert_data['actions'][training_size + validation_size:]

  if args.print_debug:
    print('training shapes:', training_x.shape, training_y.shape)
    print('validation shapes:', validation_x.shape, validation_y.shape)
    print('testing shapes:', testing_x.shape, testing_y.shape)

  # Making the neural net ##	
  # Constants
  num_features = training_x.shape[1]
  num_labels = training_y.shape[1]

  # Choosing hyper parameters
  prefix = 'experts/Roboschool'
  policy_type = args.expert_policy_file[len(prefix) : args.expert_policy_file.rfind('-v1')]
  batch_size = HYPER_PARAMETER_DICT[policy_type]['batch_size']
  learning_rate = HYPER_PARAMETER_DICT[policy_type]['learning_rate']
  num_relus = HYPER_PARAMETER_DICT[policy_type]['num_relus']
  beta_A = HYPER_PARAMETER_DICT[policy_type]['beta_A']
  beta_B = HYPER_PARAMETER_DICT[policy_type]['beta_B']
  beta_output = HYPER_PARAMETER_DICT[policy_type]['beta_output']
  num_steps = HYPER_PARAMETER_DICT[policy_type]['num_steps']

  graph = tf.Graph()
  with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(None, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(validation_x)
    tf_test_dataset = tf.constant(testing_x)

    # Variables.
    weights_A = tf.Variable(tf.truncated_normal([num_features, num_relus]))
    biases_A = tf.Variable(tf.zeros([num_relus]))
    weights_B = tf.Variable(tf.truncated_normal([num_relus, num_relus]))
    biases_B = tf.Variable(tf.zeros([num_relus]))
    weights_output = tf.Variable(tf.truncated_normal([num_relus, num_labels]))
    biases_output = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    hidden_layer_A = tf.nn.relu(tf.matmul(tf_train_dataset, weights_A) + biases_A)
    hidden_layer_B = tf.nn.relu(tf.matmul(hidden_layer_A, weights_B) + biases_B)
    logits = tf.matmul(hidden_layer_B, weights_output) + biases_output
    loss = tf.reduce_mean(tf.squared_difference(tf_train_labels, logits)) + beta_A * tf.nn.l2_loss(weights_A) + beta_B * tf.nn.l2_loss(weights_B) + beta_output * tf.nn.l2_loss(weights_output)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = logits
    valid_prediction = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_A) + biases_A), weights_B) + biases_B), weights_output) + biases_output
    test_prediction = tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_A) + biases_A), weights_B) + biases_B), weights_output) + biases_output

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    ### Training ###
    print("Starting training...")
    for i in range(args.num_rollouts):
      if args.print_debug:
        print('iter', i)
      ### Generate data via DAgger ###
      if i != 0:
        obs = env.reset()
        done = False
        steps = 0
        while not done:
          expert_action = policy.act(obs)
          policy_action = session.run(logits, feed_dict = {tf_train_dataset : obs[None, : ]})[0]
          training_x = np.append(training_x, obs.reshape(1, obs.shape[0]), axis=0)
          training_y = np.append(training_y, expert_action.reshape(1, expert_action.shape[0]), axis=0)
          obs, r, done, _ = env.step(policy_action)
          steps += 1
          if args.print_debug and steps % 100 == 0:
            print("%i/%i"%(steps, max_steps))
          if steps >= max_steps:
            break

      ### Gradient descent ###
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        offset = (step * batch_size) % (training_y.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = training_x[offset:(offset + batch_size), :]
        batch_labels = training_y[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, training_loss, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if args.print_debug and step % 1000 == 0:
          print("Minibatch loss at step %d: %f" % (step, training_loss))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validation_y))
      print("Test set accuracy: %.1f%%" % accuracy(test_prediction.eval(), testing_y))

    ### Scoring ###
    print("Starting scoring...")
    returns = []
    for i in range(args.num_rollouts):
      if args.print_debug:
        print('iter', i)
      obs = env.reset()
      done = False
      totalr = 0.
      steps = 0
      while not done:
        action = session.run(logits, feed_dict = {tf_train_dataset : obs[None, : ]})[0]
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if args.render:
          env.render()
        if args.print_debug and steps % 100 == 0:
          print("%i/%i"%(steps, max_steps))
        if steps >= max_steps:
          break
      returns.append(totalr)
    print('DAgger returns', returns)
    print('DAgger mean return', np.mean(returns))
    print('DAgger std of return', np.std(returns))
  print("DONE!!!")

if __name__ == '__main__':
  main()
