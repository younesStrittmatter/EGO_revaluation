"""
main script to compare psyneulink EGO model and python implementation
"""

from comparision.data import (get_baseline_trials, get_reward_revaluation_trials,
                              get_transition_revaluation_trials, get_time_sequence)

from comparision.utils import plot_trials

def main():
    baseline_trials, baseline_rewards = get_baseline_trials(num_seqs=20)
    reward_revaluation_trials, reward_revaluation_rewards = get_reward_revaluation_trials(num_seqs=20)
    transition_revaluation_trials, transition_revaluation_rewards = get_transition_revaluation_trials(num_seqs=20)

    assert len(reward_revaluation_trials) == len(transition_revaluation_trials)

    num_trials_absolute = len(baseline_trials) + len(reward_revaluation_trials)

    time_sequence = get_time_sequence(num_trials=num_trials_absolute)

    # plot_trials(baseline_trials, baseline_rewards)
    #
    # import comparision.experiment1 as experiment1
    # import torch_implementation.utils as utils
    #
    # params = utils.Map(
    #     n_participants=58,
    #     n_simulations=100,  # number of rollouts per participant
    #     n_steps=3,  # number of steps per rollout
    #     state_d=7,  # length of state vector
    #     context_d=7,  # length of context vector
    #     time_d=25,  # length of time vector
    #     self_excitation=.25,  # rate at which old context is carried over to new context
    #     input_weight=.45,  # rate at which state is integrated into new context
    #     retrieved_context_weight=.3,  # rate at which context retrieved from EM is integrated into new context
    #     time_noise=.01,  # noise std for time integrator (drift is set to 0)
    #     state_weight=.5,  # weight of the state used during memory retrieval
    #     context_weight=.3,  # weight of the context used during memory retrieval
    #     time_weight=.2,  # weight of the time used during memory retrieval
    #     temperature=.05,  # temperature of the softmax used during memory retrieval (smaller means more argmax-like)
    #     seed=1234  # random seed for the simulation
    # )
    #
    # visited_states_baseline, rewards_baseline = experiment1.gen_baseline_trials(params)
    # visited_states_reward_revaluation, rewards_reward_revaluation = experiment1.gen_reward_revaluation_trials(params)
    # visited_states_transition_revaluation, rewards_transition_revaluation = experiment1.gen_transition_revaluation_trials(
    #     params)
    #
    # plot_trials(visited_states_baseline, rewards_baseline)

    # print('Baseline:')
    # for bt, br in zip(baseline_trials, baseline_rewards):
    #     print(bt, br)
    # print('Reward Revaluation:')
    # for bt, br in zip(reward_revaluation_trials, reward_revaluation_rewards):
    #     print(bt, br)
    # print('Transition Revaluation:')
    # for bt, br in zip(transition_revaluation_trials, transition_revaluation_rewards):
    #     print(bt, br)



if __name__ == "__main__":
    main()

