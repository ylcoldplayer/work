import random
from enum import Enum
from numpy.random import lognormal
from src.logger import work_logger


class LambdaMode(Enum):
    FLOOR_MODE = 1
    CAP_MODE = 2
    INIT_MODE = 3


STEP_SIZE = 0.05


class OptimalLambdaSimulator:
    def __init__(self, pay_prices, total_budget, bid_floor, bid_cap, bid_init, lambda_mode):
        random.seed(0)
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

    def _get_logger_file_name(self):
        lambda_mode = 'mode_' + str(self.lambda_mode.name)
        total_budget = 'B_' + str(self.total_budget)
        bid_floor = 'bf_' + str(self.bid_floor)
        bid_cap = 'bd_' + str(self.bid_cap)
        bid_init = 'bi_' + str(self.bid_init)
        step_size = 'sz_' + str(self.step_size)

        file_name = [lambda_mode, total_budget, bid_floor, bid_cap, bid_init, step_size]
        return '_'.join(file_name) + '.log'

    def simulate(self):
        lambda_floor, lambda_cap = self._compute_lambda_floor_cap()
        T = len(self.pay_prices)
        logger_file_name = self._get_logger_file_name()
        logger = work_logger.get_work_logger('lambda_simulation', file_name=logger_file_name)
        for t in range(T):
            round_t = "round " + str(t)
            pay_price = self.pay_prices[t]
            lambda_t = self.lambda_t
            bid = self.bid_bench/self.lambda_t
            bid = min(max(self.bid_floor, bid), min(self.bid_cap, self.remaining_budget))
            logger.info(round_t + " bid price: " + str(bid))
            logger.info(round_t + " pay price: " + str(pay_price))
            logger.info(round_t + " lambda: " + str(lambda_t))
            if bid >= pay_price:
                cost_ratio = pay_price/self.expected_cost
                self.remaining_budget -= pay_price
                self.total_win += 1
                logger.info(round_t + " cost ratio: " + str(cost_ratio))
            else:
                cost_ratio = 0.
                logger.info(round_t + " cost ratio: " + str(cost_ratio))
            lambda_t = lambda_t - self.step_size*(1-cost_ratio)
            self.lambda_t = min(max(lambda_floor, lambda_t), lambda_cap)
            logger.info(round_t + " remaining budget: " + str(self.remaining_budget))
            logger.info("*********************************************************************************************")
            logger.info("*********************************************************************************************")
        logger.info('total_win: ' + str(self.total_win) + ' , remaining_budget: ' + str(self.remaining_budget))
        return self.total_win, self.remaining_budget


def generate_pay_price(mu=2., sigma=0.5):
    """
    To visualize lognormal distribution, goto https://homepage.divms.uiowa.edu/~mbognar/applets/lognormal.html
    :param mu:
    :param sigma:
    :return:
    """
    return max(0.1, lognormal(mu, sigma, 1)[0])


if __name__ == '__main__':
    random.seed(0)
    # synthesize pay prices
    pay_ps = []
    for i in range(96):
        pay_ps.append(generate_pay_price())
    B = 80
    bid_f = 0.5
    bid_c = 20.
    bid_i = 2.
    lambda_m = LambdaMode.FLOOR_MODE
    simulator = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, lambda_m)
    print(simulator.simulate())
