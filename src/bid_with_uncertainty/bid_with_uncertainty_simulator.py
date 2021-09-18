import math
import random
from enum import Enum

import numpy.random
from numpy.random import lognormal

from src.logger import work_logger
from src.bid_with_uncertainty.expert_advisor import ExpertAdvisor


class BenchmarkMode(Enum):
    FLOOR_MODE = 1
    CAP_MODE = 2
    INIT_MODE = 3
    CONST_MODE = 4


class LRDecayMode(Enum):
    FIXED = 1
    LINEAR = 2
    EXP = 3


STEP_SIZE = 0.05
CONST_BID = 100
COST_RATIO_CAP = 2.
ANNEAL_SPEED = 0.0075
STEP_SIZE_FLOOR = 0.05
DECAY_RATE = 10.
B = 50

# EXP3 algorithm const
ETA = 0.1
NUM_SLOT = 11


def get_perturbation():
    return numpy.random.normal(0, 0.03, 1)[0]


def generate_pay_price(mu=0, sigma=0.5):
    """
    To visualize lognormal distribution, goto https://homepage.divms.uiowa.edu/~mbognar/applets/lognormal.html
    :param mu:
    :param sigma:
    :return:
    """
    return max(0.1, lognormal(mu, sigma, 1)[0])


class BidWithUncertaintySimulator:
    def __init__(self, pay_prices, total_budget, bid_floor, bid_cap, bid_init, lambda_mode, step_size=STEP_SIZE,
                 no_auction_error=0., lr_mode=LRDecayMode.FIXED, with_noisy_traffic=False):
        random.seed(0)
        self.pay_prices = pay_prices
        self.total_budget = total_budget
        self.bid_floor = bid_floor
        self.bid_cap = bid_cap
        self.bid_init = bid_init
        self.lambda_mode = lambda_mode
        self.lambda_t = self._compute_init_lambda()
        self.bid_benchmark = self._get_bid_benchmark()
        self.total_win = 0
        self.expected_cost = total_budget * 1. / (len(pay_prices) * (1 + no_auction_error))
        self.remaining_budget = total_budget
        self.step_size = step_size
        self.lr_mode = lr_mode
        self.expert = ExpertAdvisor(eta=ETA, num_slot=NUM_SLOT)
        self.with_noisy_traffic = with_noisy_traffic

    def _compute_init_lambda(self):
        if self.lambda_mode is BenchmarkMode.FLOOR_MODE:
            return self.bid_floor * 1. / self.bid_init
        elif self.lambda_mode is BenchmarkMode.CAP_MODE:
            return self.bid_cap * 1. / self.bid_init
        else:
            return 1.

    def _get_bid_benchmark(self):
        if self.lambda_mode is BenchmarkMode.FLOOR_MODE:
            return self.bid_floor
        elif self.lambda_mode is BenchmarkMode.CAP_MODE:
            return self.bid_cap
        else:
            return self.bid_init

    def _compute_lambda_floor_cap(self):
        if self.lambda_mode is BenchmarkMode.FLOOR_MODE:
            return self.bid_floor / self.bid_cap, 1.
        elif self.lambda_mode is BenchmarkMode.CAP_MODE:
            return 1., self.bid_cap / self.bid_floor
        else:
            return self.bid_init / self.bid_cap, self.bid_init / self.bid_floor

    def _get_logger_file_name(self):
        lambda_mode = 'mode_' + str(self.lambda_mode.name)
        total_budget = 'B_' + str(self.total_budget)
        bid_floor = 'bf_' + str(self.bid_floor)
        bid_cap = 'bc_' + str(self.bid_cap)
        bid_init = 'bi_' + str(self.bid_init)
        step_size = 'sz_' + str(self.step_size)

        file_name = [lambda_mode, total_budget, bid_floor, bid_cap, bid_init, step_size]
        return '_'.join(file_name) + '.log'

    def simulate(self):
        lambda_floor, lambda_cap = self._compute_lambda_floor_cap()
        T = len(self.pay_prices)
        logger_file_name = self._get_logger_file_name()
        logger = work_logger.get_work_logger('lambda_simulation', file_name=logger_file_name)

        bid_records = []
        budget_consumption_records = []

        for t in range(T):
            round_t = "round " + str(t)
            pay_price = self.pay_prices[t]
            lambda_t = self.lambda_t
            bid = self.bid_benchmark / self.lambda_t
            bid = min(max(self.bid_floor, bid), min(self.bid_cap, self.remaining_budget))
            bid_records.append(bid)
            if self.lambda_mode is BenchmarkMode.CONST_MODE:
                bid = min(self.remaining_budget, CONST_BID)
            logger.info(round_t + " bid price: " + str(bid))
            logger.info(round_t + " pay price: " + str(pay_price))
            logger.info(round_t + " lambda: " + str(lambda_t))
            if bid >= pay_price:
                # projected traffic is actual traffic with pertrubation
                projected_traffic = T * (1 + get_perturbation())
                projected_expected_cost = T / projected_traffic
                cost_ratio = pay_price / projected_expected_cost
                # logic using Exp4 to adjust traffic
                if self.with_noisy_traffic:
                    suggested_traffic = T / self.expert.get_action_multiplier(projected=projected_traffic, actual=T)
                    suggested_expected_cost = self.total_budget / suggested_traffic
                    cost_ratio = pay_price / suggested_expected_cost
                cost_ratio = min(cost_ratio, COST_RATIO_CAP)
                self.remaining_budget -= pay_price
                self.total_win += 1
                logger.info(round_t + " cost ratio: " + str(cost_ratio))

            else:
                cost_ratio = 0.
                logger.info(round_t + " cost ratio: " + str(cost_ratio))

            budget_consumption_records.append(B - self.remaining_budget)
            sz = self.step_size
            if self.lr_mode == LRDecayMode.LINEAR:
                sz = max(self.step_size - t * ANNEAL_SPEED, STEP_SIZE_FLOOR)
            elif self.lr_mode == LRDecayMode.FIXED:
                sz = max(self.step_size * math.exp(-DECAY_RATE * t), STEP_SIZE_FLOOR)
            lambda_t = lambda_t - sz * (1 - cost_ratio)
            self.lambda_t = min(max(lambda_floor, lambda_t), lambda_cap)
            logger.info(round_t + " remaining budget: " + str(self.remaining_budget))
            logger.info("*********************************************************************************************")
            logger.info("*********************************************************************************************")
        logger.info('total_win: ' + str(self.total_win) + ' , remaining_budget: ' + str(self.remaining_budget))
        return self.total_win, self.remaining_budget, bid_records, budget_consumption_records


if __name__ == "__main__":
    B = 50
    bid_f = 0.1
    bid_c = 10.0
    bid_i = 2.0

    pay_ps = []
    for _ in range(144):
        pay_ps.append(generate_pay_price())
    simulator = BidWithUncertaintySimulator(pay_prices=pay_ps, total_budget=B, bid_floor=bid_f, bid_cap=bid_c,
                                            bid_init=bid_i, lambda_mode=BenchmarkMode.FLOOR_MODE, step_size=0.05,
                                            with_noisy_traffic=False)
    simulator.simulate()
