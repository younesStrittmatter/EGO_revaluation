# Change Set

## Time Vector

In the original implementation, the time vector is initialized and updated within the memory itself. Here, we
generate the time vector externally and pass it to the memory update.

Also the time vector update, and retreval function has been changed. In the original version, the time vector
is initialized as small vector:

```python
time_code = torch.zeros((params['time_d'],),dtype=torch.float)+.01
```

and updated by adding a small noise value:

```python
time_code += torch.randn_like(time_code)*time_noise
```

this is equiavalent to a positive drift rate of `time_noise/2` plus a noise with standard deviation of 
`sqrt(time_noise/2)` (uniform).

However, later the time_vector is normalized both in memory and as query. Since the vector is always positive
and always growing, this means the larger the time value, the closer proceding vectors will be to each other after
normalization (which is unintended).

Instead, we use DriftOnASphere that implements a Brownian motion on a unit sphere. This means the time vector is
always normalized (length 1) but the difference between two proceding time vectors it independent of their absoulte
value (as intended).

We use PsyNeuLink's `DriftOnASphere` function to implement this.