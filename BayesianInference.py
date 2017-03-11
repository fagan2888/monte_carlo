class Prior(object):
    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Implement this method!")


class Likelihood(object):
    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Implement this method!")


class Proposal(object):
    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Implement this method!")

    def is_symmetric(self):
        raise NotImplementedError("Implement this method!")

    def transition_prob(self, x, y):
        raise NotImplementedError("Implement this method!")


class BayesianInference(object):
    def __init__(self, prior, likelihood, observations, proposal=None):
        self.prior = prior
        self.likelihood = likelihood
        self.observations = observations
        self.proposal = proposal
        self.mcmc_chain = []
        self.prob_chain = []
        self.prop_chain = []
        self.iterations = 0
        self.accepted = 0

    def reset(self):
        self.mcmc_chain = []
        self.prob_chain = []
        self.prop_chain = []
        self.iterations = 0
        self.accepted = 0

    def sample(self, n, p0=None, seed=None):
        """ Create a fresh sample where the initial point is p0
        :type p0: type accepted by prior/likelihood etc
        :type seed: int
        :param p0: initial point
        """
        import numpy.random as rn
        if seed is not None:
            rn.seed(seed)
        if p0 is not None:
            self.reset()
            self.mcmc_chain.append(p0)
            self.prob_chain.append(self.prior_prob(p0) * self.likelihood_prob(p0))
            self.sample(n)
        else:
            for _ in range(n):
                self.iterations += 1
                cur_state, cur_prob = self.mcmc_chain[-1], self.prob_chain[-1]
                proposal, prob = self.make_proposal(cur_state)
                self.prop_chain.append(proposal)
                accepted = False
                if prob > cur_prob:
                    accepted = True
                else:
                    u = rn.rand()
                    if self.proposal.is_symmetric():
                        if u < prob / cur_prob:
                            accepted = True
                    else:
                        (g_xy, g_yx) = self.proposal.transition_prob(cur_state, proposal)
                        if u < (prob * g_xy) / (cur_prob * g_yx):
                            accepted = True
                if accepted:
                    self.accepted += 1
                    self.mcmc_chain.append(proposal)
                    self.prob_chain.append(prob)
                else:
                    self.mcmc_chain.append(cur_state)
                    self.prob_chain.append(cur_prob)

    def prior_prob(self, state):
        return self.prior(state)

    def likelihood_prob(self, state):
        return self.likelihood(state, self.observations)

    def make_proposal(self, cur_state):
        proposal = self.proposal(cur_state)
        prob = self.prior_prob(proposal) * self.likelihood_prob(proposal)
        return proposal, prob

    def take_samples(self, burn=0, thin=1):
        """ Take samples from the chain """
        if burn == 0:
            burn = int(len(self.mcmc_chain) * 0.1)
        return self.mcmc_chain[burn::thin]


