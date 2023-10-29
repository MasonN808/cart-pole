import math
import numpy as np

class CartPoleEnv():
    
    def __init__(self):
        # Constants
        self.g = 9.8    # Gravity
        self.m_c = 1    # cart mass
        self.m_p = .1   # pole mass
        self.m_t = self.m_c + self.m_p  # Total Mass
        self.l = .5     # pole length
        self.tau = .02  # action rate (action per second)

        # s = (x, v, ω,  ω'), where x ∈ R is the horizontal position of the cart along the track, v ∈ R is the cart
        # velocity, ω ∈ R is the pole angle (in radians), and  ω' ∈ R is the pole’s angular velocity
        self.obs = {
            "x": 0,
            "vx": 0,
            "w": 0,
            "vw": 0
        }

        self.steps = 0 # Actions performed
        self.done = False # Terminal state reached

    def reset(self):
        # Constants
        self.g = 9.8    # Gravity
        self.m_c = 1    # cart mass
        self.m_p = .1   # pole mass
        self.m_t = self.m_c + self.m_p  # Total Mass
        self.l = .5     # pole length
        self.tau = .02  # action rate (action per second)

        # s = (x, v, ω,  ω'), where x ∈ R is the horizontal position of the cart along the track, v ∈ R is the cart
        # velocity, ω ∈ R is the pole angle (in radians), and  ω' ∈ R is the pole’s angular velocity
        self.obs = {
            "x": 0,
            "vx": 0,
            "w": 0,
            "vw": 0
        }

        self.steps = 0 # Actions performed
        self.done = False # Terminal state reached
    
    def _reward(self, action: float):
        return 1
    
    def _next_obs(self, action: float, obs: dict):
        # Extracting the pole angle ω_t and pole angular velocity ω'_t from the current observation
        x = obs["x"]
        vx = obs["vx"]
        w = obs["w"]
        vw = obs["vw"]
        F = action

        # Calculate b based on the updated equation provided
        b = (F + self.m_p * self.l * vw**2 * math.sin(w)) / self.m_t

        # Calculate c based on the equation provided
        numerator_c = self.g * math.sin(w) - math.cos(w) * b
        denominator_c = self.l * (4/3 - (self.m_p * math.cos(w)**2) / self.m_t)
        c = numerator_c / denominator_c

        # Calculate d based on the equation provided
        d = b - (self.m_p * self.l * c * math.cos(w) / self.m_t)

        # Update the state based on the provided equations
        x_next = x + self.tau * vx
        vx_next = vx + self.tau * d
        w_next = w + self.tau * vw
        vw_next = vw + self.tau * c

        # Update the observation dictionary with the new state values
        obs["x"] = x_next
        obs["vx"] = vx_next
        obs["w"] = w_next
        obs["vw"] = vw_next

        return obs
    

    def _done(self, obs: dict):
        if obs["w"] < -math.pi/15 or obs["w"] > math.pi/15:
            # Pole fell
            return True
        if obs["x"] < -2.4 or obs["x"] > 2.4:
            # Cart reached border of env
            return True
        if self.steps > 500:
            # From timeout
            return True
        return False
    
    def step(self, action: float):
        obs = self._next_obs(action, self.obs)
        self.obs = obs      # This may not be the best place to put this
        reward = self._reward(action)
        done = self._done(obs)
        self.done = done
        self.steps += 1
        return obs, reward, done

