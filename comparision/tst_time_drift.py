import torch
from comparision.data import get_time_sequence

time_noise = .01
n_trials = 120

TIME_D = 25

def norm(key):
    return key/key.norm(dim=-1,keepdim=True)

def get_time_code_original():
    times = []
    time_code = torch.zeros((TIME_D,), dtype=torch.float) + .01

    for _ in range(n_trials):
        time_code += torch.randn_like(time_code) * time_noise

        times.append(norm(time_code.clone()))
    return times

def main():
    times = get_time_sequence(
        num_trials=n_trials,
    )

    times = [torch.asarray(t) for t in times]

    # check that all time codes are normalized
    print('checking normalization of time codes:')
    for i, t in enumerate(times):
        norm_t = torch.norm(t).item()
        print(f"Time code {i} norm: {norm_t:.6f}")

    print()

    # Print the differences between consecutive time codes to observe drift
    print('differences between consecutive time codes:')
    for i in range(1, len(times)):
        diff = torch.norm(times[i] - times[i-1]).item()
        print(f"Difference between time code {i} and {i-1}: {diff:.6f}")

    print('***')
    print()
    # print moving average of differences
    diffs = [torch.norm(times[i] - times[i-1]).item() for
                i in range(1, len(times))]
    window_size = 10
    moving_averages = []
    for i in range(len(diffs) - window_size + 1):
        window = diffs[i:i + window_size]
        moving_average = sum(window) / window_size
        moving_averages.append(moving_average)

    print("\nMoving averages of differences (window size = 10):")
    for i, ma in enumerate(moving_averages):
        print(f"From trial {i+1} to {i+window_size}: {ma:.6f}")

    # cosine similarities between consecutive time codes
    print()
    print('cosine similarities between consecutive time codes:')
    for i in range(1, len(times)):
        cos_sim = torch.dot(times[i], times[i-1]).item()
        print(f"Cosine similarity between time code {i} and {i-1}: {cos_sim:.6f}")

    # cosine similarities as moving average of cosine similarities
    cos_sims = [torch.dot(times[i], times[i-1]).item() for
                i in range(1, len(times))]
    moving_averages_cos = []
    print()
    print("\nMoving averages of cosine similarities (window size = 10):")
    for i in range(len(cos_sims) - window_size + 1):
        window = cos_sims[i:i + window_size]
        moving_average = sum(window) / window_size
        moving_averages_cos.append(moving_average)

    for i, ma in enumerate(moving_averages_cos):
        print(f"From trial {i+1} to {i+window_size}: {ma:.6f}")



if __name__ == "__main__":
    main()







