import math

from numpy.random import choice, normal

INCREMENT = 0.01


class ExpertAdvisor:
    """
    This class implements Exp3 algorithm in adversarial environment to advise the bidder to choose the cost ratio for
    simulation.
    """

    def __init__(self, eta, num_slot):
        # learning rate eta
        self.eta = eta
        self.num_slot = num_slot
        self.weights = [1] * num_slot

        self.accurate_slot = (self.num_slot - 1) / 2
        self.relative_slot_max = math.floor(self.num_slot / 2)
        self.relative_slot_min = - self.relative_slot_max
        self.first_round = True

    def get_action_multiplier(self, projected, actual):
        if self.first_round:
            self.first_round = False
            return 1.0
        action_slot = self.update_weights(projected, actual)
        return 1.0 + action_slot * INCREMENT

    def update_weights(self, projected, actual):
        """
        Update weights using Exp3 algorithm
        :param projected: projected traffic
        :param actual: actual traffic
        :return: the action slot
        """
        p_distribution = self._compute_distribution()
        action_slot = self._sample_from_distribution()
        index = action_slot + self.relative_slot_max
        observed_reward = self._compute_reward(projected, actual)
        estimated_reward = observed_reward / p_distribution[index]
        self.weights[index] *= math.exp(self.eta/self.num_slot * estimated_reward)
        return action_slot

    def _sample_from_distribution(self):
        indices = []
        for i in range(self.num_slot):
            indices.append(i - self.relative_slot_max)
        p_dist = self._compute_distribution()
        return choice(indices, size=1, p=p_dist)[0]

    def _compute_distribution(self):
        total_weight = sum(self.weights)
        normalized_weights = [w/total_weight for w in self.weights]
        return [(1-self.eta)*w + self.eta/self.num_slot for w in normalized_weights]

    def _compute_reward(self, projected, actual):
        diff_max = INCREMENT * ((self.num_slot-1)/2)
        relative_diff = min(abs((projected - actual) / actual), diff_max)
        relative_accuracy = relative_diff / diff_max
        return 1 - relative_accuracy


if __name__ == "__main__":
    print(choice([1, 2, 3], size=1, p=[0.1, 0.5, 0.4]))
    print(normal(0, 0.03, 1))