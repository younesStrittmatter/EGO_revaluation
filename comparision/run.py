"""
main scripts to run the experiments
"""
from functools import partial
from typing import Callable

from comparision.data import (get_baseline_trials, get_reward_revaluation_trials,
                              get_transition_revaluation_trials, get_time_sequence)

import comparision.model_original as model_original

import comparision.params as params
import comparision.utils as utils

import torch
import numpy as np


def run(
        estimate_reward_from_starting_state: Callable,
        num_participants: int = params.N_PARTICIPANTS,
        num_seqs_baseline: int = params.N_BASELINE_TRIALS,
        num_seqs_revaluation: int = params.N_REVALUATION_TRIALS,
        n_simulations: int = params.N_SIMULATIONS,  # number of simulation trajectories
        n_steps: int = params.N_STEPS,  # number of steps per simulation trajectory
        state_retrieval_weight: float = params.STATE_RETRIEVAL_WEIGHT,
        context_retrieval_weight: float = params.CONTEXT_RETRIEVAL_WEIGHT,
        time_retrieval_weight: float = params.TIME_RETRIEVAL_WEIGHT,
        old_context_integration_rate: float = params.OLD_CONTEXT_INTEGRATION_RATE,
        state_integration_rate: float = params.STATE_INTEGRATION_RATE,
        new_context_integration_rate: float = params.NEW_CONTEXT_INTEGRATION_RATE,
        context_d=params.STATE_SIZE,
        state_d=params.STATE_SIZE,
        time_d=params.TIME_SIZE,

):
    # initialize revaluation scores
    revaluation_scores = np.zeros((num_participants, 3))
    for participant_idx in range(num_participants):
        # ** GENERATE TRIALS ** #

        # Baseline phase
        baseline_trials, baseline_rewards = get_baseline_trials(
            num_seqs=num_seqs_baseline)

        # Reward revaluation phase

        # get only the revaluation trials
        reward_revaluation_trials_only, reward_revaluation_rewards_only = get_reward_revaluation_trials(
            num_seqs=num_seqs_revaluation)

        # combine with baseline trials
        reward_revaluation_trials = np.concatenate(
            [baseline_trials, reward_revaluation_trials_only], axis=0)
        reward_revaluation_rewards = np.concatenate(
            [baseline_rewards, reward_revaluation_rewards_only], axis=0)

        # Transition revaluation phase

        # get only the revaluation trials
        transition_revaluation_trials_only, transition_revaluation_rewards_only = get_transition_revaluation_trials(
            num_seqs=num_seqs_revaluation)

        # combine with baseline trials
        transition_revaluation_trials = np.concatenate(
            [baseline_trials, transition_revaluation_trials_only], axis=0)
        transition_revaluation_rewards = np.concatenate(
            [baseline_rewards, transition_revaluation_rewards_only], axis=0)

        # Time code for all trials
        time_sequence = get_time_sequence(
            num_trials=len(reward_revaluation_trials)
        )

        # ** GENERATE MEMORIES ** #

        # convenience partial function to generate memories with fixed parameters
        _gen_memories = partial(
            model_original.gen_memories,
            old_context_integration_rate=old_context_integration_rate,
            state_integration_rate=state_integration_rate,
            retrieved_context_integration_rate=new_context_integration_rate,
            state_retrieval_weight=state_retrieval_weight,
            context_retrieval_weight=context_retrieval_weight,
            time_retrieval_weight=time_retrieval_weight,
            context_d=context_d)

        # memories baseline only
        memories_baseline = _gen_memories(
            visited_states=baseline_trials,
            rewards=baseline_rewards,
            time_sequence=time_sequence[:len(baseline_trials)],  # only the time codes for baseline trials
        )

        # memories reward revaluation
        memories_reward_reval = _gen_memories(
            visited_states=reward_revaluation_trials,
            rewards=reward_revaluation_rewards,
            time_sequence=time_sequence,  # all time codes
        )

        # memories transition revaluation
        memories_transition_reval = _gen_memories(
            visited_states=transition_revaluation_trials,
            rewards=transition_revaluation_rewards,
            time_sequence=time_sequence,  # all time codes
        )

        # ** ESTIMATE REWARDS FROM STARTING STATES ** #
        starting_state_1 = torch.eye(7)[1]
        starting_state_2 = torch.eye(7)[2]

        _estimated_reward_from_starting_state = partial(
            estimate_reward_from_starting_state,
            n_simulations=n_simulations,
            n_steps=n_steps,
            state_retrieval_weight=state_retrieval_weight,
            context_retrieval_weight=context_retrieval_weight,
            time_retrieval_weight=time_retrieval_weight,
            old_context_integration_rate=old_context_integration_rate,
            state_integration_rate=state_integration_rate,
            new_context_integration_rate=new_context_integration_rate,
            context_d=context_d,
            state_d=state_d,
            time_d=time_d
        )

        estimated_reward_state_1_baseline = _estimated_reward_from_starting_state(
            memories=memories_baseline,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_baseline = _estimated_reward_from_starting_state(
            memories=memories_baseline,
            starting_state=starting_state_2)

        estimated_reward_state_1_reward_reval = _estimated_reward_from_starting_state(
            memories=memories_reward_reval,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_reward_reval = _estimated_reward_from_starting_state(
            memories=memories_reward_reval,
            starting_state=starting_state_2)

        estimated_reward_state_1_transition_reval = _estimated_reward_from_starting_state(
            memories=memories_transition_reval,
            starting_state=starting_state_1,
        )

        estimated_reward_state_2_transition_reval = _estimated_reward_from_starting_state(
            memories=memories_transition_reval,
            starting_state=starting_state_2)

        state_one_preference_baseline = estimated_reward_state_1_baseline - estimated_reward_state_2_baseline
        state_one_preference_reward_reval = estimated_reward_state_1_reward_reval - estimated_reward_state_2_reward_reval
        state_one_preference_transition_reval = estimated_reward_state_1_transition_reval - estimated_reward_state_2_transition_reval

        revaluation_scores[participant_idx, 0] = state_one_preference_baseline - state_one_preference_reward_reval
        revaluation_scores[participant_idx, 1] = state_one_preference_baseline - state_one_preference_transition_reval
        revaluation_scores[participant_idx, 2] = state_one_preference_baseline

    return revaluation_scores


run_original = partial(
    run,
    estimate_reward_from_starting_state=model_original.estimate_reward_from_starting_state
)

# run_original()
#
#
# def run_experiment(params):
#     revaluation_scores = np.zeros((params['n_participants'], 3))
#     for participant_idx in range(params['n_participants']):
#         visited_states_baseline, rewards_baseline = gen_baseline_trials(params)
#         memories = gen_memories(visited_states_baseline, rewards_baseline, params)
#         estimated_reward_state_one_baseline = estimate_reward_from_starting_state(memories, torch.eye(7)[1], params)
#         estimated_reward_state_two_baseline = estimate_reward_from_starting_state(memories, torch.eye(7)[2], params)
#
#         visited_states_reward_reval, rewards_reward_reval = gen_reward_revaluation_trials(params)
#         visited_states_reward_reval = torch.cat([visited_states_baseline, visited_states_reward_reval], axis=0)
#         rewards_reward_reval = torch.cat([rewards_baseline, rewards_reward_reval], axis=0)
#         memories = gen_memories(visited_states_reward_reval, rewards_reward_reval, params)
#         estimated_reward_state_one_reward_reval = estimate_reward_from_starting_state(memories, torch.eye(7)[1], params)
#         estimated_reward_state_two_reward_reval = estimate_reward_from_starting_state(memories, torch.eye(7)[2], params)
#
#         visited_states_transition_reval, rewards_transition_reval = gen_transition_revaluation_trials(params)
#         visited_states_transition_reval = torch.cat([visited_states_baseline, visited_states_transition_reval], axis=0)
#         rewards_transition_reval = torch.cat([rewards_baseline, rewards_transition_reval], axis=0)
#         memories = gen_memories(visited_states_transition_reval, rewards_transition_reval, params)
#         estimated_reward_state_one_transition_reval = estimate_reward_from_starting_state(memories, torch.eye(7)[1],
#                                                                                           params)
#         estimated_reward_state_two_transition_reval = estimate_reward_from_starting_state(memories, torch.eye(7)[2],
#                                                                                           params)
#
#         visited_states_control, rewards_control = gen_control_trials(params)
#         visited_states_control = torch.cat([visited_states_baseline, visited_states_control], axis=0)
#         rewards_control = torch.cat([rewards_baseline, rewards_control], axis=0)
#         memories = gen_memories(visited_states_control, rewards_control, params)
#         estimated_reward_state_one_control = estimate_reward_from_starting_state(memories, torch.eye(7)[1], params)
#         estimated_reward_state_two_control = estimate_reward_from_starting_state(memories, torch.eye(7)[2], params)
#
#         state_one_preference_baseline = estimated_reward_state_one_baseline - estimated_reward_state_two_baseline
#         state_one_preference_reward_reval = estimated_reward_state_one_reward_reval - estimated_reward_state_two_reward_reval
#         state_one_preference_transition_reval = estimated_reward_state_one_transition_reval - estimated_reward_state_two_transition_reval
#         state_one_preference_control = estimated_reward_state_one_control - estimated_reward_state_two_control
#
#         revaluation_scores[participant_idx, 0] = state_one_preference_baseline - state_one_preference_reward_reval
#         revaluation_scores[participant_idx, 1] = state_one_preference_baseline - state_one_preference_transition_reval
#         revaluation_scores[participant_idx, 2] = state_one_preference_baseline - state_one_preference_control
#     return revaluation_scores
#
#
# def main():
#     baseline_trials, baseline_rewards = get_baseline_trials(num_seqs=20)
#     reward_revaluation_trials, reward_revaluation_rewards = get_reward_revaluation_trials(num_seqs=20)
#     transition_revaluation_trials, transition_revaluation_rewards = get_transition_revaluation_trials(num_seqs=20)
#
#     assert len(reward_revaluation_trials) == len(transition_revaluation_trials)
#
#     num_trials_absolute = len(baseline_trials) + len(reward_revaluation_trials)
#
#     time_sequence = get_time_sequence(num_trials=num_trials_absolute)
#
#     # plot_trials(baseline_trials, baseline_rewards)
#     #
#     # import comparision.experiment1 as experiment1
#     # import torch_implementation.utils as utils
#     #
#     # params = utils.Map(
#     #     n_participants=58,
#     #     n_simulations=100,  # number of rollouts per participant
#     #     n_steps=3,  # number of steps per rollout
#     #     state_d=7,  # length of state vector
#     #     context_d=7,  # length of context vector
#     #     time_d=25,  # length of time vector
#     #     self_excitation=.25,  # rate at which old context is carried over to new context
#     #     input_weight=.45,  # rate at which state is integrated into new context
#     #     retrieved_context_weight=.3,  # rate at which context retrieved from EM is integrated into new context
#     #     time_noise=.01,  # noise std for time integrator (drift is set to 0)
#     #     state_weight=.5,  # weight of the state used during memory retrieval
#     #     context_weight=.3,  # weight of the context used during memory retrieval
#     #     time_weight=.2,  # weight of the time used during memory retrieval
#     #     temperature=.05,  # temperature of the softmax used during memory retrieval (smaller means more argmax-like)
#     #     seed=1234  # random seed for the simulation
#     # )
#     #
#     # visited_states_baseline, rewards_baseline = experiment1.gen_baseline_trials(params)
#     # visited_states_reward_revaluation, rewards_reward_revaluation = experiment1.gen_reward_revaluation_trials(params)
#     # visited_states_transition_revaluation, rewards_transition_revaluation = experiment1.gen_transition_revaluation_trials(
#     #     params)
#     #
#     # plot_trials(visited_states_baseline, rewards_baseline)
#
#     # print('Baseline:')
#     # for bt, br in zip(baseline_trials, baseline_rewards):
#     #     print(bt, br)
#     # print('Reward Revaluation:')
#     # for bt, br in zip(reward_revaluation_trials, reward_revaluation_rewards):
#     #     print(bt, br)
#     # print('Transition Revaluation:')
#     # for bt, br in zip(transition_revaluation_trials, transition_revaluation_rewards):
#     #     print(bt, br)
#
#
# if __name__ == "__main__":
#     main()
