import torch
import numpy as np

import scipy

class PPOBuffer():
    """
    "A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs."
    """
    def __init__(self, obs_dim:int, act_dim:int, size:int|None=None, gamma:float=0.99, lam:float=0.95) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma # discounting factor
        self.lam = lam # lambda: Generalized Advantage Estimation (GAE) factor

        self.idx = 0 # current buffer index
        self.path_start_idx = 0 # index of current path start

        self.initialized = False
        
    def set_size(self, size:int) -> None:
        """QoL Function that allows to specify the buffer size independently from buffer creation.
        Used due to my fix ideas about what part of the learning/ training should go to the Agent and what part to the trainer
        Args:
            size (int): size of the buffer
        """
        def combined_shape(a, b) -> tuple:
            if b is None or b ==1:
                return (a,)
            return (a, b) if np.isscalar(b) else (a, *b)
        
        # basic buffered attributes:
        self.obs_buf = np.zeros(combined_shape(size, self.obs_dim), dtype=np.float32) # observation buffer
        self.act_buf = np.zeros(combined_shape(size, self.act_dim), dtype=np.float32) # action buffer
        self.rew_buf = np.zeros(size, dtype=np.float32) # reward buffer

        # estimated by ACtorCritic Agent:
        self.val_buf = np.zeros(size, dtype=np.float32) # value of state buffer
        self.log_prob_buf = np.zeros(size, dtype=np.float32) # logarithmic probability of action a buffer

        # caclucated here and stored for convenience:
        self.ret_buf = np.zeros(size, dtype=np.float32) # return buffer?
        self.adv_buf = np.zeros(size, dtype=np.float32) # advantage buffer
        
        self.max_size = size
        self.initialized = True

    def store(self, obs, act, rew, val, log_prob) -> None:
        assert (self.initialized)
        assert (self.idx < self.max_size)
        self.obs_buf[self.idx] = obs
        self.act_buf[self.idx] = act
        self.rew_buf[self.idx] = rew
        self.val_buf[self.idx] = val
        self.log_prob_buf[self.idx] = log_prob

        self.idx += 1

    def finish_path_estimates(self, last_value:float=0.):
        assert (self.initialized)
        path_slice = slice(self.path_start_idx, self.idx)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)

        # do GAE estimate of q-value/ advantage
        # GAE = weighted average of all n-step return Q-Value estiamtes
        # weight depends on horizon n

        q_1_step_estimates = rews[:-1] + self.gamma * vals[1:]
        adv_1_step_estimates = q_1_step_estimates - vals[:-1]

        # # estimate GAE for action a_t as
        # # A_GAE(a_t) = sum _k=t^N (\gamma \lambda)^(k-t) * A_n(a_k)
        # # where A_k = advantage computed with above 1-step return and stored in adv_1_step_estimates
        # # understandable, but inefficient implementation:
        # res = []
        # for i in range(len(adv_1_step_estimates)):
        #     r = 0
        #     j = i
        #     while j < len(adv_1_step_estimates):
        #         r += self.gamma*self.lam**(j-i) * adv_1_step_estimates[j]
        #         j += 1 
        #     res.append(r)
        # adv_gae_estimates = np.array(res)

        # magix, but efficient implementation:
        def discount_cumsum(x, discount):
            """
            magic from rllab for computing discounted cumulative sums of vectors.
            * invert order of x with x[::-1]
            * see notes of https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
            input: 
                vector x = [x0, x1, x2]
            output:
                [x0 + discount * x1 + discount^2 * x2,  x1 + discount * x2, x2]
            """
            return scipy.signal.lfilter(
                    [1], 
                    [1, float(-discount)], 
                    x[::-1],
                    axis=0
                )[::-1]
        adv_gae_estimates = discount_cumsum(adv_1_step_estimates, self.gamma * self.lam)

        self.adv_buf[path_slice] = adv_gae_estimates

        # discount rewards to get estimated returns per state-action pair. These are the targets for the Value estimator
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # reset pointer to the end of the path
        self.path_start_idx = self.idx

    def get(self, normalize:bool=True) -> dict:
        """Get buffer content with advantage normalized (var=1) around zero
        Returns:
            dict: obs, act, ret, adv, lop_prob
        """
        assert self.idx == self.max_size
        # after getting the content, we prepare the buffer to be filled again:
        self.idx, self.path_start_idx = 0, 0

        # normalize advantage:
        if normalize:
            adv_mean = self.adv_buf.mean()
            adv_std  = self.adv_buf.std()
            self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return_data = dict(
            obs=self.obs_buf, act=self.act_buf, 
            ret=self.ret_buf, adv=self.adv_buf,
            log_prob=self.log_prob_buf
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in return_data.items()}
    
    
