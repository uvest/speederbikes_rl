# Speederbike Reinforcement Learning

This repo contains a collection of RL algorithm implementations on the [Speekerbikes simulation environment]().

## Training

For local training you can tweak the parameters of the `training_XX.py` script of the respective algorithm and simply run it with, e.g.

```bash
python train_ddqn.py
```

be aware that you need to set up your environment with the required packages.

### Training on HPC
For training on an HPC cluster you can use the respective `training_XX_hpc.sh` scripts.

#### HPC information

1. Login
    login nodes listed at: https://www.hrz.tu-darmstadt.de/hlr/betrieb_hlr/hardware_hlr_1/aktuell_verfuegbar_hlr/index.en.jsp

    To log in choose on of the addresses there for ssh. e.g. `ssh <TU-ID>@lcluster14.hrz.tu-darmstadt.de` and your TU-ID password.
2. Create a connection to gitlab to pull your code.
   * `ssh-keygen`: The usual
   * Pull respective project
3. Starting a batch job
   * ...

Notes regarding usage of Login notes:
* "While test-driving your software on a login node, check its current CPU load with “top” or “uptime”, and reduce your impact by using less cores/threads/processes or at least by using “nice”."

## DDQN - Double Deep Q-Learning Network
A most simple model-free Temporal Difference/Q-Learning algorithm based on Hasselt et al. (2015).

We work with two networks:
* target network $Q_t$
* online/ local network $Q_l$

The online/ local network is updated according to a temporal differnce error, a MSE between the current prediction using the local network $Q_l$ and a *target* value $Q^*$computed using both, the target and the lcoal network:

$$error_t = (Q^*(s_t, a_t) - Q_l(s_t, a_t))^2$$

where the target

$$Q^*(s_t, a_t) = r_t + \gamma Q_t(s_{t+1}, \argmax_{a'}(Q_l(s_{t+1}, a')))$$

Then the target network parameters $\theta_t$ is updated using Polyak averaging

$$\theta_t \gets \tau \theta_l + (1-\tau) \theta_t$$

## PPO - Proximal Policy Optimization
A more complicated on-policy Policy gradient algorithm that 
* uses importance sampling
* limits the policy update by an approximated KL divergence
* uses a Genralized Advantage Estimate to estimate the Advantage used for the target estimate

Fills the experience buffer until it is full. Then does updates until the new policy diverged too far from the old one.


## LQG - Linear Quadratic Regulator
A baseline control algorithm not based on learning but on linear, known system dynamcis and a quadratic cost function.

