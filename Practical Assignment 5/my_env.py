import numpy as np


class EvidenceEnv(object):
    """
    Very simple task which only requires evaluating present evidence and does not require evidence integration.
    The actor gets a reward when it correctly decides on the ground truth. Ground truth 0/1 determines probabilistically
    the number of 0s or 1s as observations
    """

    def __init__(self, n=1, p=0.8):
        """

        Args:
            n: number of inputs (pieces of evidence)
            p: probability of emitting the right sensation at the input
        """

        self.n_input = n
        self.p = p
        self.n_action = 2

        self._state = None

    def reset(self):
        """
        Resets state and generates new observations

        Returns:
            observation
        """

        # generate state
        self._state = np.random.choice(2)

        return self.observe()

    def step(self, action):
        """
        Executes action, updates state and returns an observation, reward, done (episodic tasks) and optional information

        :param action:
        :return: observation, reward, done, info
        """

        # return 1 for correct decision and -1 for incorrect decision
        reward = (2 * (action == self._state) - 1)

        # generate state
        self._state = np.random.choice(2)

        # we are always done after each decision
        done = True

        return self.observe(), reward, done, None

    def observe(self):
        """
        Helper function which generates an observation based on a state

        :return: observation
        """

        # generate associated observations
        P = [self.p, 1 - self.p] if self._state == 0 else [1 - self.p, self.p]

        return np.random.choice(2, self.n_input, True, P).astype('float32').reshape([1, self.n_input])[0]

    def render(self):
        """
        Takes care of rendering

        :return:
        """

        # print("State: " + str(self._state))

    def close(self):
        """
        Closes the rendering

        :return:
        """
        pass

    def asint(self,obs):
        """
        Represent input observations as an integer number
        :param obs:
        :return:
        """
        return int(sum(2**i*b for i, b in enumerate(obs)))

    def asbinary(self, i, b_len):
        """
        Represent integer as binary array
        :param i: integer
        :param b_len: length of binary array
        :return:
        """

        # get binary representation from integer
        _b = [int(x) for x in list('{0:0b}'.format(i))]
        _b = [0 for _ in range(b_len - len(_b))] + _b

        return _b

    def toBinary(self, obs):
        intValue = self.asint(obs)
        return self.asbinary(intValue, 2)
