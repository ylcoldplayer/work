import random
from enum import Enum
from numpy.random import lognormal


class LambdaMode(Enum):
    FLOOR_MODE = 1
    CAP_MODE = 2
    INIT_MODE =3


STEP_SIZE = 0.1


class OptimalLambdaSimulator:
    def __init__(self, pay_prices, total_budget, bid_floor, bid_cap, bid_init, lambda_mode):
        self.pay_prices = pay_prices
        self.total_budget = total_budget
        self.bid_floor = bid_floor
        self.bid_cap = bid_cap
        self.bid_init = bid_init
        self.lambda_mode = lambda_mode
        self.lambda_t = self._compute_init_lambda()
        self.bid_bench = self._get_bid_bench()
        self.total_win = 0
        self.expected_cost = total_budget*1./len(pay_prices)
        self.remaining_budget = total_budget
        self.step_size = STEP_SIZE

    def _compute_init_lambda(self):
        if self.lambda_mode is LambdaMode.FLOOR_MODE:
            return self.bid_floor*1./self.bid_init
        elif self.lambda_mode is LambdaMode.CAP_MODE:
            return self.bid_cap*1./self.bid_init
        else:
            return 1.

    def _get_bid_bench(self):
        if self.lambda_mode is LambdaMode.FLOOR_MODE:
            return self.bid_floor
        elif self.lambda_mode is LambdaMode.CAP_MODE:
            return self.bid_cap
        else:
            return self.bid_init

    def _compute_lambda_floor_cap(self):
        if self.lambda_mode is LambdaMode.FLOOR_MODE:
            return self.bid_floor/self.bid_cap, 1.
        elif self.lambda_mode is LambdaMode.CAP_MODE:
            return 1., self.bid_cap/self.bid_floor
        else:
            return self.bid_init/self.bid_cap, self.bid_init/self.bid_floor

    def simulate(self):
        lambda_floor, lambda_cap = self._compute_lambda_floor_cap()
        T = len(self.pay_prices)
        for _ in range(T):
            pay_price = self.pay_prices[i]
            lambda_t = self.lambda_t
            bid = self.bid_bench/self.lambda_t
            bid = min(max(self.bid_floor, bid), min(self.bid_cap, self.remaining_budget))
            if bid >= pay_price:
                cost_ratio = pay_price/self.expected_cost
                self.remaining_budget -= pay_price
                self.total_win += 1
            else:
                cost_ratio = 0.
            lambda_t = lambda_t - self.step_size*(1-cost_ratio)
            self.lambda_t = min(max(lambda_floor, lambda_t), lambda_cap)
        return self.total_win, self.remaining_budget


def generate_pay_price(mu=2., sigma=0.5):
    return max(0.1, lognormal(mu, sigma, 1)[0])


if __name__ == '__main__':
    random.seed(0)
    # synthesize pay prices
    pay_prices = []
    for i in range(96):
        pay_prices.append(generate_pay_price())
    B = 80
    bid_f = 0.5
    bid_c = 20.
    bid_i = 5.
    lambda_m = LambdaMode.INIT_MODE
    simulator = OptimalLambdaSimulator(pay_prices, B, bid_f, bid_c, bid_i, lambda_m)
    print(simulator.simulate())
