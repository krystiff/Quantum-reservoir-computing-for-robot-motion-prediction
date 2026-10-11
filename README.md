# Quantum reservoir computing for robot motion prediction

A trapped-ion-style quantum reservoir computer used as the memory of a robot controller. The reservoir is
benchmarked against classical baselines on a robot manipulation task, and the trained controller runs as
Qiskit circuits on Qollab's simulators. Press **Run** to replay a held-out human demonstration through the
trained controller; *How to run it* and *What to expect* below explain the output.

Everything here is offline prediction: the controller is scored on how closely it reproduces a human
operator's recorded actions from the recorded observations. It was not run in the robot simulator, so its
predictions never change what it sees next, and nothing here shows that it lifts the cube.

**Team:** Marcus Demierre, Juan Rodriguez Hernandez and Krystifer Campos.

This effort is supported by Qollab & IonQ.

## Introduction

### Quantum reservoir computing

Reservoir computing is a way to learn from time-dependent data, such as time series, signals or control tasks. The input sequence drives a fixed dynamical system, the *reservoir*, and we only train a simple linear *readout* on top of its state. Classically, the reservoir is a random recurrent neural network called an Echo State Network (ESN). A *quantum* reservoir swaps that network for a set of interacting qubits and uses their measured expectation values as features.

First, n qubits live in a 2ⁿ-dimensional state space, so a handful of them might do the job of a much larger classical network ([Fujii and Nakajima, 2017](https://arxiv.org/abs/1602.08159), found that 5 to 7 qubits could rival recurrent networks of 100 to 500 nodes). Second, the quantum part is never trained, so there's no costly circuit optimisation. Third, the dissipation that usually hurts quantum hardware becomes useful here: it gives the reservoir its fading memory. Whether all this adds up to a real advantage is still an open question, and that's what this project sets out to test.

### The task: imitating a robot arm

We use the **Lift** task from the robomimic benchmark, where a Franka Panda robot arm picks up a cube from a table. The dataset (`low_dim_v15.hdf5`) has 200 demonstrations recorded by a human operator; this project uses the first 100 of them. At each time step, 19 numbers describe the scene: where the gripper is and how it's oriented, how open its fingers are, and where the cube is relative to the gripper. The target is the operator's action at that step, 7 numbers that say how to move and rotate the gripper and whether to open or close it. In short, the model watches the states so far and must guess what the human did next.

### Why quantum reservoirs for this dataset?

This task is a natural fit for reservoir computing. The right action depends not just on the current state but on where we are in the motion (approaching, grasping or lifting), so temporal context may help. The mapping is also partly nonlinear: the gripper command, for example, flips suddenly from "open" to "close" once the gripper reaches the cube, a jump that a weighted sum of the inputs can't reproduce. And with only 19 inputs, the problem is small enough to encode into a few qubits, which makes it a manageable proof of concept for quantum-circuit simulation.

### The reservoirs and the readout

At every time step of a demonstration, the quantum reservoir does this:

1. The 19 observations are standardised and projected to one rotation angle per qubit (RY).
2. Every qubit gets an RZ rotation, and every pair of qubits an XX rotation with a fixed random strength.
3. Every qubit loses a fraction 0.3 of its excitation (amplitude damping), which makes old inputs fade. The
   damping is built into the circuit with ancilla qubits; it is a design choice, not a model of hardware noise.
4. The averages of Z and X on every qubit, and of ZZ and XX on every pair, are the features for the readout.

The state is carried from one time step to the next within a demonstration. The classical reservoir is a
leaky echo state network: tanh nodes with fixed random connections, spectral radius 1.0 and leak rate 0.3.

For both, only the linear readout is trained, in two ways: with Adam (200 epochs, keeping the epoch with the
lowest validation loss) and by ridge regression (strength chosen on the validation demonstrations). The 100
demonstrations are split 70/15/15 into training, validation and test sets, and a demonstration never
straddles two sets. A linear model on the current observation is the reference, and the trained controller
also runs as circuits.

## Results

Test loss is the mean squared error against the human's actions on 15 held-out demonstrations (lower is better).

| Model | Features | Adam readout | Exact (ridge) readout |
|---|---|---|---|
| Classical reservoir, 10 nodes | 10 | 0.0565 | 0.0564 |
| Classical reservoir, 20 nodes | 20 | 0.0481 | 0.0478 |
| Classical reservoir, 50 nodes | 50 | 0.0459 | 0.0456 |
| Classical reservoir, 100 nodes | 100 | 0.0376 | 0.0370 |
| Quantum reservoir, 4 qubits | 20 | 0.0985 | 0.0834 |
| Quantum reservoir, 6 qubits | 42 | 0.0605 | 0.0590 |
| Quantum reservoir, 8 qubits | 72 | 0.0521 | 0.0483 |
| Quantum reservoir, 10 qubits | 110 | 0.0503 | 0.0453 |
| Quantum reservoir, 12 qubits | 156 | 0.0474 | 0.0432 |

For reference, always predicting the average action gives 0.1740 and a linear model on the current
observation gives 0.0448.

![Test error against reservoir size](src/results/sweep/size_sweep.png)

- The quantum reservoir improves steadily with size and reaches the linear model at 10 to 12 qubits.
- The 100-node classical reservoir is clearly ahead. We found no quantum advantage on this task.
- The policy in the Qollab demo gives its readout the current observation as well as the reservoir's
  features. That lowers the linear model's 0.0448 to 0.0408. Trained on features with simulated measurement
  noise it scores 0.0430, and 0.0439 when that noise is also added to the test features. The noise is an
  independent Gaussian approximation to 1,000 shots; no circuits were run for these two numbers.

Limits of these numbers: one random reservoir per size and 15 test demonstrations, so the significance of
small gaps between rows has not been established. The Adam readout has not fully converged for the quantum reservoirs. The error is measured
against the human's actions; the policy was not run in the robot simulator, so there is no success rate.

## Repository structure

### On Qollab 

What Qollab runs needs only NumPy and Qiskit. The main files in more detail:

| File | What it does |
|---|---|
| `main.py` | replays one held-out demonstration with the trained 6-qubit policy, first by exact simulation, then as circuits on the selected backend |
| `lift_policy.npz` | the trained policy (input weights, couplings, damping, readout) and the demonstration it replays |
| `src/train_readout.py` | trains that policy from the 6-qubit reservoir of `sweep.py`, with a readout made robust to measurement shot noise, and saves `lift_policy.npz` |
| `src/sweep.py` | trains the classical and quantum reservoir readouts at every size; resumable, results go to `src/results/sweep/results.json` |
| `src/reservoirs_numpy.py` | the echo state network and the density-matrix quantum reservoir, in NumPy |
| `src/utils_numpy.py` | feature extraction, prediction and metrics, in NumPy |
| `src/video_data.py` | loading, validation, split, normalisation and metrics for the demonstration data |

### On your own computer

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## How to run it

### On Qollab

Open the project, choose the Built-in simulator or an IonQ simulator, and press **Run**. Nothing has to be
installed or configured. The Run settings are optional:

| Setting | Default | What it does |
|---|---|---|
| `shots` | 500 | how many times each circuit is run; the readout was trained for the noise of 1,000 shots |
| `replay` | 6 | how many past frames each circuit replays to rebuild the reservoir's memory (2 on IonQ backends) |
| `every` | 8 | every 8th frame of the demonstration is measured on the backend |
| `retrain` | off | also trains the reservoirs and the policy again inside the run |
| `max_qubits` | 6 | the largest quantum reservoir that `retrain` trains |

### On your own computer

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## What to expect

### From the Qollab run

A default run on the Built-in simulator prints this (shortened):

```
Demonstration with 53 frames, 6 reservoir qubits, damping 0.3
Exact reservoir policy vs expert on this demonstration: MSE 0.0411, MAE 0.1108

Running 12 circuits of 7 qubits on the backend ...
Backend: 500 shots per circuit, last 6 frames replayed per circuit
frame    action   expert    exact  same circuit  measured
    7    move z    -0.48    -0.58         -0.58     -0.61
    7   gripper    -1.00    -1.30         -1.32     -1.27
  ...
   47    move z     0.81     0.46          0.46      0.46
   47   gripper     1.00     0.73          0.74      0.84
Measured vs exact simulation of the same circuits, mean squared difference: 0.0022
Replaying 6 frames vs the full history, mean squared difference: 0.0000

Benchmark. What was run: src/sweep.py (NumPy, offline) with these settings
  ... the settings block of src/sweep.py

What it gave (stored results): test MSE and MAE against the expert on the 15 held-out demonstrations
model                           readout  features      selected      MSE      MAE  seconds
average action                                  0                 0.1740   0.2406
linear on observation                          19                 0.0448   0.1185
classical reservoir, 10 nodes      adam        10     epoch 177   0.0565   0.1322        0
  ... one row per reservoir size and readout
quantum reservoir, 12 qubits      ridge       156  ridge 0.0001   0.0432   0.1110     5261
Trained parameters: lift_policy.npz for the policy above, src/results/sweep/<run>_<readout>.npz for each row
```

The `measured` column and the `Measured vs exact` line change from run to run, because each circuit is sampled
500 times. In our runs that line was between 0.001 and 0.003. Everything else is the same on every run.

| Output | What it means |
|---|---|
| `Exact reservoir policy vs expert ...` | the policy, simulated exactly, against the human on this demonstration, which is the held-out demonstration with the median error. Commands run from −1 to 1, so it is off by about 0.11 on average |
| `Running 12 circuits of 7 qubits ...` | six frames are measured, each with one circuit in the Z basis and one in the X basis. A circuit has the 6 reservoir qubits and one ancilla for the damping |
| `expert` | what the human did; `move z` is up and down, and for `gripper` −1 is open and +1 is closed |
| `exact` | the policy, simulated exactly with the full history |
| `same circuit` | an exact simulation of the shortened circuit that was sent |
| `measured` | the policy recomputed from the counts measured on the backend |
| `Measured vs exact simulation of the same circuits ...` | how far the backend is from an exact simulation. On the Built-in simulator that is shot noise only; the IonQ simulators also apply a noise model of the device |
| `Replaying 6 frames vs the full history ...` | what is lost by replaying only the last 6 frames: about 0.00003 for the frames of this demonstration, which prints as 0.0000 |
| `Benchmark. What was run ...` | the settings of the offline benchmark, printed from `src/sweep.py` itself |
| `What it gave (stored results) ...` | the stored benchmark: the Results table with MAE, the selected epoch or ridge strength, and the offline run time |

The figure has two halves. The upper half shows the seven commands along the demonstration: the human in
grey, the exact policy in blue, and the actions recomputed from the measured circuits as orange dots. The
lower half is the size-sweep figure of the Results section.

![Upper half of the figure of a default run on the Built-in simulator](src/results/qollab_run.png)

On an IonQ simulator the run sends a single circuit of 24 qubits that uses fresh ancillas in place of reset,
because those backends accept one job per run. The table then has one frame (frame 31), each panel has one
orange dot, and the last line reads `Replaying 2 frames vs the full history, mean squared difference: 0.0016`.
We ran it on the IonQ Forte 1 simulator, which applies IonQ's noise model of that device: the job took 39
seconds, and `Measured vs exact` read 0.0012 and 0.0031 in two runs.

With `retrain` ticked the run also prints `Retraining 6 reservoirs now ...` with the seconds per reservoir
(about 7 seconds in total up to 6 qubits), the results table again for the retrained models, and
`Largest difference between the retrained and the stored test MSE: 5.6e-16`. It then retrains the policy of
the demo and prints `Largest difference between its readout weights and the ones this run started with:
1.8e-14`. Black rings in the figure mark the retrained points.

### From the scripts

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## Extend it

- **Other sizes, seeds or settings.** Every setting written as a list in `ESN_CONFIG` or `QUANTUM_CONFIG` at
  the top of `src/sweep.py` is swept, for example `"num_qubits": [4, 6, 8]` or `"seed": [42, 1, 2]`. Use a
  new `RESULTS_DIR` after changing a setting that is not swept.
- **Other observables or couplings.** `single_observables`, `pair_observables` and `interaction` (`"XX"`,
  `"YY"` or `"ZZ"`) are settings of `QUANTUM_CONFIG`. The Qollab circuits are written for the default
  reservoir, and `train_readout.py` checks for it.
- **Another task.** Save its demonstrations in the format of `final_dataset.pkl` (see *Dataset files*) and
  point `DATASET_PATH` at it. The numbers of observations and actions are read from the data.
- **A success rate.** The policy was not run in the robot simulator. Rolling it out in robosuite would turn
  the error against the human into a success rate.

The rest of this page documents the data and the training code in detail.

## Input Data

### Source: robomimic Lift (Proficient-Human), `low_dim_v15.hdf5`

The data comes from robomimic, a framework and benchmark for robot learning from demonstration developed by the Stanford Vision and Learning Lab as part of the ARISE Initiative. 

| | |
|---|---|
| Library | robomimic, <https://github.com/ARISE-Initiative/robomimic> |
| Dataset documentation | <https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html> |
| Task | **Lift**: a Franka Panda robotic arm grasps a small cube on a table and lifts it |
| Demonstrator type | **Proficient-Human (PH)**: one experienced operator teleoperating the robot via RoboTurk; 200 successful demonstrations |
| File used | `lift/ph/low_dim_v15.hdf5` (low-dimensional observations, no images) |
| File size | ~21 MB |

**Reference.** If you use this dataset, please cite the original robomimic paper:

> A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín, "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," *Conference on Robot Learning (CoRL)*, 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)

**Version note.** The demonstrations were recorded in the robosuite simulator (MuJoCo physics), environment version 1.5.1, as stored in the file's env_args metadata. This is why the file is named `low_dim_v15.hdf5` rather than the `low_dim_v141.hdf5` listed on the v0.3 documentation page. Neither robomimic nor robosuite is a runtime dependency of this project: they were used only to obtain and inspect the original file, and the pipeline reads it directly with h5py.


### Licensing

| Component | Role in this project | License |
|---|---|---|
| robomimic (ARISE-Initiative/robomimic) | Source of the Lift PH dataset and its format | MIT |
| robosuite (ARISE-Initiative/robosuite) | Simulator that generated the demonstrations | MIT |

### Original file structure (HDF5)

Besides arrays, HDF5 groups can carry small **metadata attributes** (marked "metadata:" in the tree). They describe the group itself rather than holding training data. This project only reads `env_args`, to confirm the task is `"Lift"`.

```
low_dim_v15.hdf5
├── data/                      metadata: environment settings (env_args, JSON text),
│   │                                    total time steps across all demos (total = 9666)
│   ├── demo_0/                metadata: time steps in this demo (num_samples = T),
│   │   │                                MuJoCo scene description (model_file, XML text)
│   │   ├── obs/               observations at time t
│   │   │   ├── robot0_eef_pos          (T, 3)
│   │   │   ├── robot0_eef_quat         (T, 4)
│   │   │   ├── robot0_gripper_qpos     (T, 2)
│   │   │   ├── object                  (T, 10)
│   │   │   ├── robot0_eef_quat_site    (T, 4)    ┐
│   │   │   ├── robot0_gripper_qvel     (T, 2)    │
│   │   │   ├── robot0_joint_pos        (T, 7)    │ present in the file,
│   │   │   ├── robot0_joint_pos_cos    (T, 7)    │ NOT used in this project
│   │   │   ├── robot0_joint_pos_sin    (T, 7)    │
│   │   │   └── robot0_joint_vel        (T, 7)    ┘
│   │   ├── next_obs/          same keys, shifted by one step (obs at t+1)
│   │   ├── actions            (T, 7)
│   │   ├── rewards            (T,)
│   │   ├── dones              (T,)
│   │   └── states             (T, 32)   full MuJoCo simulator state
│   ├── demo_1/
│   ├── ...
│   └── demo_199/
└── mask/                      predefined train/validation splits (lists of demo names)
    ├── train (180)            valid (20)
    ├── 50_percent (100)       50_percent_train (90)   50_percent_valid (10)
    └── 20_percent (40)        20_percent_train (36)   20_percent_valid (4)
```

Each demo is an independent episode with its own length `T`. The listing below shows a few demos and the arrays used by this project:

```
demo_1:   58 time steps | robot0_eef_pos (58, 3), robot0_eef_quat (58, 4), robot0_gripper_qpos (58, 2), object (58, 10), actions (58, 7)
demo_10:  54 time steps | robot0_eef_pos (54, 3), robot0_eef_quat (54, 4), robot0_gripper_qpos (54, 2), object (54, 10), actions (54, 7)
demo_100: 47 time steps | robot0_eef_pos (47, 3), robot0_eef_quat (47, 4), robot0_gripper_qpos (47, 2), object (47, 10), actions (47, 7)
demo_101: 43 time steps | robot0_eef_pos (43, 3), robot0_eef_quat (43, 4), robot0_gripper_qpos (43, 2), object (43, 10), actions (43, 7)
demo_102: 58 time steps | robot0_eef_pos (58, 3), robot0_eef_quat (58, 4), robot0_gripper_qpos (58, 2), object (58, 10), actions (58, 7)
```

### Variables used

#### Inputs: 19 features per time step (robot and object state)

| HDF5 key | Dim | Column names | Meaning |
|---|---|---|---|
| `robot0_eef_pos` | 3 | `eef_x, eef_y, eef_z` | End-effector (gripper) position in the world frame [m] |
| `robot0_eef_quat` | 4 | `eef_qx, eef_qy, eef_qz, eef_qw` | End-effector orientation as a unit quaternion (x, y, z, w) |
| `robot0_gripper_qpos` | 2 | `finger_1, finger_2` | Positions of the two finger joints [m]. They are approximately mirror images (`finger_2 ≈ −finger_1`); values range from about 0.02 to 0.04 in magnitude |
| `object[0:3]` | 3 | `cube_x, cube_y, cube_z` | Cube position [m] |
| `object[3:7]` | 4 | `cube_qx, cube_qy, cube_qz, cube_qw` | Cube orientation as a quaternion |
| `object[7:10]` | 3 | `grip2cube_x, grip2cube_y, grip2cube_z` | Relative vector from the gripper to the cube. Checked numerically to equal `cube_pos − eef_pos` exactly |

#### Targets: 7 values per time step (the operator's action)

| Column | Meaning | Range in the data |
|---|---|---|
| `act_dx, act_dy, act_dz` | Commanded end-effector translation, normalized to [−1, 1] (±1 maps to ±5 cm per step) | [−1, 1] |
| `act_drx, act_dry, act_drz` | Commanded end-effector rotation (axis-angle), normalized to [−1, 1] (±1 maps to ±0.5 rad per step) | about [−0.39, 0.47] in the 100 demonstrations used |
| `act_grip` | Gripper command: −1 = open, +1 = close | exactly {−1, +1} (binary) |

The learning problem is therefore **behavior cloning as sequence regression**: given the state sequence up to time *t*, predict the action the human took at time *t*. The gripper target is binary, while the other six are continuous.

**Redundancy note.** Some inputs are redundant: `grip2cube_*` is an exact linear combination of `cube_*` and `eef_*`, and the two finger coordinates are nearly mirror images. This means the 19 inputs carry fewer than 19 independent degrees of freedom. 

### From HDF5 to nested lists (`dataset_generation.ipynb`)

The notebook `dataset_generation.ipynb` (needs `h5py`, `numpy`, `pandas`) turns the HDF5 file into plain Python nested lists, keeping each demo at its own length with no padding. It loads the first 100 of the 200 Lift demos (`N_DEMOS = 100`), joins the four observation groups into 19 inputs per time step, and pairs each step with the 7-value action the operator took at that same step. After checking that every demo is consistent (matching lengths, no missing values), it saves the lists, column names and demo names to a pickle file (`final_dataset.pkl`, next to the HDF5 file).

#### Resulting structure

```
inputs  = [video_1, video_2, ..., video_N]        # one element per demo (N = 100)
video_i = [time_steps, values]
    time_steps = [1, 2, ..., T_i]                 # T_i in [36, 59], differs per demo
    values     = [[19 floats at step 1],
                  [19 floats at step 2],
                  ...]                            # T_i rows

targets = [video_1, ..., video_N]                 # same structure, 7 floats per step

input_keys  = ["eef_x", "eef_y", "eef_z", "eef_qx", ..., "grip2cube_z"]   # 19 names
target_keys = ["act_dx", "act_dy", "act_dz", "act_drx", "act_dry", "act_drz", "act_grip"]
demo_names  = ["demo_0", "demo_1", ...]
```

Example (`demo_0`, 59 steps, first time step):

```
t = 1
inputs : [-0.0905, -0.0152, 1.0118, 0.9972, -0.0072, 0.0740, 0.0019, 0.0208, -0.0208,
           0.0264,  0.0270, 0.8314, 0.0000,  0.0000, 0.9691, 0.2466, 0.1169,  0.0422, -0.1804]
targets: [-0.0, 0.0, 0.0, 0.0038, 0.1482, 0.0145, -1.0]
```

## Video inputs and decision targets

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## Train and export

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## Evaluate with NumPy only

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).

## Reservoir state and streaming

See the [GitHub repository](https://github.com/juanfrh7/Quantum-reservoir-computing-for-robot-motion-prediction).



## Credits and disclosure

- **Data:** the Lift demonstrations were recorded in the robosuite simulator and published as part of robomimic (Mandlekar et al., CoRL 2021). `final_dataset.pkl` and the demonstration stored in `lift_policy.npz` are derived from them. See *Input Data* above for the full reference and licences.
- **Libraries:** Qiskit, NumPy, Matplotlib and PyTorch.
- **Background:** Quantum reservoir computing was introduced by Fujii and Nakajima, "Harnessing
  Disordered-Ensemble Quantum Dynamics for Machine Learning", Phys. Rev. Applied 8, 024030 (2017).
- **AI assistance:** AI tools (Claude and ChatGPT) helped write and debug code and draft parts of this
  write-up. The team designed the project, ran every experiment and checked the results.
- **Build window:** All code here was written during the hackathon, 9 to 11 October 2026. 



The code is released under the MIT licence (see `LICENSE`).


