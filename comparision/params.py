STATE_SIZE = 7  # length of state vector (number of unique states)
CONTEXT_SIZE = STATE_SIZE  # length of the context vector (usually the same as state vector)
TIME_SIZE = 25  # length of the time vector (arbitrary choice)

# MODEL parameters

# retrieval weight (weight the matches by these weights when retrieving from memory)
STATE_RETRIEVAL_WEIGHT = .5
CONTEXT_RETRIEVAL_WEIGHT = .3
TIME_RETRIEVAL_WEIGHT = .2

# integration rates (how much to integrate old context, state and retrieved context into the new context)
OLD_CONTEXT_INTEGRATION_RATE = .25
STATE_INTEGRATION_RATE = .45
NEW_CONTEXT_INTEGRATION_RATE = .3


TEMPERATURE = .05 # temperature of the softmax used during memory retrieval (smaller means more argmax-like)

# SIMULATION parameters
N_PARTICIPANTS = 200 #58  # number of participants to simulate
N_BASELINE_TRIALS = 20  # number of baseline trials per participant (one sequence = one trial of each stimulus sequence)
N_REVALUATION_TRIALS = 20 # number of revaluation trials per participant (one sequence = one trial of each stimulus sequence)

N_SIMULATIONS = 1  # number of rollouts per participant
N_STEPS = 3  # number of steps per rollout

REWARD_BASELINE_1 = 1
REWARD_BASELINE_2 = 0.01