import random
from enum import Enum

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from numpy.random import lognormal
from src.logger import work_logger


class BenchmarkMode(Enum):
    FLOOR_MODE = 1
    CAP_MODE = 2
    INIT_MODE = 3
    CONST_MODE = 4


STEP_SIZE = 0.05
CONST_BID = 100


class OptimalLambdaSimulator:
    def __init__(self, pay_prices, total_budget, bid_floor, bid_cap, bid_init, lambda_mode, step_size=STEP_SIZE):
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
        self.expected_cost = total_budget*1./len(pay_prices)
        self.remaining_budget = total_budget
        self.step_size = step_size

    def _compute_init_lambda(self):
        if self.lambda_mode is BenchmarkMode.FLOOR_MODE:
            return self.bid_floor*1./self.bid_init
        elif self.lambda_mode is BenchmarkMode.CAP_MODE:
            return self.bid_cap*1./self.bid_init
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
            return self.bid_floor/self.bid_cap, 1.
        elif self.lambda_mode is BenchmarkMode.CAP_MODE:
            return 1., self.bid_cap/self.bid_floor
        else:
            return self.bid_init/self.bid_cap, self.bid_init/self.bid_floor

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
                cost_ratio = pay_price/self.expected_cost
                self.remaining_budget -= pay_price
                self.total_win += 1
                logger.info(round_t + " cost ratio: " + str(cost_ratio))

            else:
                cost_ratio = 0.
                logger.info(round_t + " cost ratio: " + str(cost_ratio))

            budget_consumption_records.append(B-self.remaining_budget)
            lambda_t = lambda_t - self.step_size*(1-cost_ratio)
            self.lambda_t = min(max(lambda_floor, lambda_t), lambda_cap)
            logger.info(round_t + " remaining budget: " + str(self.remaining_budget))
            logger.info("*********************************************************************************************")
            logger.info("*********************************************************************************************")
        logger.info('total_win: ' + str(self.total_win) + ' , remaining_budget: ' + str(self.remaining_budget))
        return self.total_win, self.remaining_budget, bid_records, budget_consumption_records


def generate_pay_price(mu=0, sigma=0.5):
    """
    To visualize lognormal distribution, goto https://homepage.divms.uiowa.edu/~mbognar/applets/lognormal.html
    :param mu:
    :param sigma:
    :return:
    """
    return max(0.1, lognormal(mu, sigma, 1)[0])


def cpi(total_win, B, remain_budget):
    if total_win == 0:
        return 0.0
    return round((B-remain_budget)/total_win, 5)


if __name__ == '__main__':
    ############################################################################################################
    ################## Uncomment the following code to test different initial bids ############################
    ############################################################################################################
    random.seed(0)
    # synthesize pay prices
    pay_ps = []
    for i in range(200):
        pay_ps.append(generate_pay_price())


    B = 50
    bid_f = 0.1
    bid_c = 10.
    bid_i = 10

    bid_inits = np.linspace(start=bid_f, stop=bid_c, num=100)

    wins_init_records = []
    wins_floor_records = []
    cpi_init_records = []
    cpi_floor_records = []

    for bid_i in bid_inits:
        simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE, step_size=0.11)
        simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE, step_size=0.05)
        wins_init, remain_budget_init, _, _ = simulator_init.simulate()
        wins_floor, remain_budget_floor, _, _ = simulator_floor.simulate()
        cpi_init = cpi(wins_init, B, remain_budget_init)
        cpi_floor = cpi(wins_floor, B, remain_budget_floor)

        wins_init_records.append(wins_init)
        wins_floor_records.append(wins_floor)
        cpi_init_records.append(cpi_init)
        cpi_floor_records.append(cpi_floor)

    fig, axe = plt.subplots(1, 2)

    axe[0].set_xlabel('initial bid')
    axe[0].set_ylabel('total wins')
    axe[0].plot(bid_inits, wins_init_records, label="InitMode")
    axe[0].plot(bid_inits, wins_floor_records, label="FloorMode")
    axe[0].legend()
    # axe[0].set_title('total_wins')

    axe[1].set_xlabel('initial bid')
    axe[1].set_ylabel('cpi')
    axe[1].plot(bid_inits, cpi_init_records, label='InitMode')
    axe[1].plot(bid_inits, cpi_floor_records, label='FloorMode')
    axe[1].legend()
    # axe[1].set_title('cpi')

    fig.suptitle('B_50_bfloor_0.1_bcap_10_init_stepsz_0.075_floor_stepsz_0.05')
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.show()





    ############################################################################################################
    ################## Uncomment the following code to simulate const bids dynamics ############################
    ############################################################################################################
    # results : max win:  78  , bid price:  0.8292929292929292 , remaining budget:  1.1868963777467685 , cpi:  0.62581

    # random.seed(0)
    # # synthesize pay prices
    # pay_ps = []
    # for i in range(200):
    #     pay_ps.append(generate_pay_price())
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 7.5
    #
    # cpi_records = []
    # win_records = []
    # remain_budget_records = []
    # const_bids = np.linspace(start=bid_f, stop=2, num=100)
    #
    # for i in range(len(const_bids)):
    #     CONST_BID = const_bids[i]
    #     simulator = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.CONST_MODE)
    #     wins, remain_budget, _, _ = simulator.simulate()
    #     cpi_tmp = cpi(wins, B, remain_budget)
    #     cpi_records.append(cpi_tmp)
    #     win_records.append(wins)
    #     remain_budget_records.append(remain_budget)
    # print(cpi_records)
    # print(win_records)
    #
    # max_idx = np.argmax(win_records)
    # print("max win: ", win_records[max_idx], " , bid price: ", const_bids[max_idx],
    #       ", remaining budget: ", remain_budget_records[max_idx], ", cpi: ", cpi_records[max_idx])
    #
    # fig, axe = plt.subplots(1, 3)
    #
    # axe[0].set_xlabel('const bid')
    # axe[0].set_ylabel('cpi')
    # axe[0].plot(const_bids, cpi_records)
    # # axe[0].legend()
    # axe[0].set_title('cpi_vs_const_bid')
    #
    # axe[1].set_xlabel('const bid')
    # axe[1].set_ylabel('total wins')
    # axe[1].plot(const_bids, win_records)
    # # axe[1].legend()
    # axe[1].set_title('total_wins_vs_const_bid')
    #
    # axe[2].set_xlabel('const bid')
    # axe[2].set_ylabel('remain budget')
    # axe[2].plot(const_bids, remain_budget_records)
    # # axe[1].legend()
    # axe[2].set_title('remain_budget_vs_const_bid')
    #
    # fig.suptitle('Results with const bids')
    # plt.subplots_adjust(hspace=0, wspace=0.5)
    # plt.show()



    ############################################################################################################
    ################## Uncomment the following code to simulate spend dynamics ############################
    ############################################################################################################
    # random.seed(0)
    # # synthesize pay prices
    # pay_ps = []
    # for i in range(200):
    #     pay_ps.append(generate_pay_price())
    #
    # fig, axe = plt.subplots(2, 2)
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 0.2
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # _, _ , _, rb_floor = simulator_floor.simulate()
    # _, _ , _, rb_init = simulator_init.simulate()
    # t_index = list(range(len(rb_floor)))
    #
    # axe[0, 0].set_ylabel('spend')
    # axe[0, 0].plot(t_index, rb_floor, label="FloorMode" )
    # axe[0, 0].plot(t_index, rb_init, label='InitMode')
    # axe[0, 0].legend()
    # axe[0, 0].set_title('binit_0.2')
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 0.5
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # _, _ , _, rb_floor = simulator_floor.simulate()
    # _, _ , _, rb_init = simulator_init.simulate()
    # t_index = list(range(len(rb_floor)))
    #
    # axe[1, 0].plot(t_index, rb_floor, label="FloorMode" )
    # axe[1, 0].plot(t_index, rb_init, label='InitMode')
    # axe[1, 0].legend()
    # axe[1, 0].set_title('binit_0.5')
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 1
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # _, _ , _, rb_floor = simulator_floor.simulate()
    # _, _ , _, rb_init = simulator_init.simulate()
    # t_index = list(range(len(rb_floor)))
    #
    # axe[0, 1].set_xlabel('time')
    # axe[0, 1].set_ylabel('Spend')
    # axe[0, 1].plot(t_index, rb_floor, label="FloorMode" )
    # axe[0, 1].plot(t_index, rb_init, label='InitMode')
    # axe[0, 1].legend()
    # axe[0, 1].set_title('binit_1')
    #
    #
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 2
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # _, _ , _, rb_floor = simulator_floor.simulate()
    # _, _ , _, rb_init = simulator_init.simulate()
    # t_index = list(range(len(rb_floor)))
    #
    # axe[1, 1].plot(t_index, rb_floor, label="FloorMode" )
    # axe[1, 1].plot(t_index, rb_init, label='InitMode')
    # axe[1, 1].legend()
    # axe[1, 1].set_title('binit_2')
    #
    #
    # fig.tight_layout()
    # plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1., wspace=0.2)
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle('B_50_bfloor_0.1_bcap_10_stepsz_0.05_Spend')
    # plt.show()


    ############################################################################################################
    ################## Uncomment the following code to simulate bid prices dynamics ############################
    ############################################################################################################
    # random.seed(0)
    # # synthesize pay prices
    # pay_ps = []
    # for i in range(200):
    #     pay_ps.append(generate_pay_price())
    #
    # fig, axe = plt.subplots(3, 2)
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 0.2
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init , _= simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # # axe[0, 0].set_xlabel('Auction time step')
    # axe[0, 0].set_ylabel('Bid price')
    # axe[0, 0].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[0, 0].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init)+ '_cpi_' + str(cpi_init))
    # axe[0, 0].legend()
    # axe[0, 0].set_title('binit_0.2')
    # # plt.show()
    #
    #
    #
    # plt.subplots(1, 2)
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 0.5
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init, _ = simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # # axe[0, 1].set_xlabel('Auction time step')
    # # axe[0, 1].set_ylabel('Bid price')
    # axe[0, 1].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[0, 1].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init)+ '_cpi_' + str(cpi_init))
    # axe[0, 1].legend()
    # axe[0, 1].set_title('binit_0.5')
    #
    #
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 1.
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init, _ = simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # # axe[1, 0].set_xlabel('Auction time step')
    # axe[1, 0].set_ylabel('Bid price')
    # axe[1, 0].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[1, 0].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init)+ '_cpi_' + str(cpi_init))
    # axe[1, 0].legend()
    # axe[1, 0].set_title('binit_1.0')
    #
    #
    #
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 2.
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init, _ = simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # # axe[1, 1].set_xlabel('Auction time step')
    # # axe[1, 1].set_ylabel('Bid price')
    # axe[1, 1].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[1, 1].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init)+ '_cpi_' + str(cpi_init))
    # axe[1, 1].legend()
    # axe[1, 1].set_title('binit_2.0')
    #
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 5.
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init, _ = simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # axe[2, 0].set_xlabel('Auction time step')
    # axe[2, 0].set_ylabel('Bid price')
    # axe[2, 0].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[2, 0].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init)+ '_cpi_' + str(cpi_init))
    # axe[2, 0].legend()
    # axe[2, 0].set_title('binit_5.0')
    #
    #
    #
    #
    #
    # B = 50
    # bid_f = 0.1
    # bid_c = 10.
    # bid_i = 10.
    #
    # simulator_floor = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.FLOOR_MODE)
    # simulator_init = OptimalLambdaSimulator(pay_ps, B, bid_f, bid_c, bid_i, BenchmarkMode.INIT_MODE)
    # win_floor, remain_budget_floor, bid_records_floor, _ = simulator_floor.simulate()
    # win_init, remain_budget_init, bid_records_init, _ = simulator_init.simulate()
    #
    # np_bid_records_floor = np.array(bid_records_floor)
    # np_bid_records_init = np.array(bid_records_init)
    # # print(bid_records_floor)
    # # print(bid_records_init)
    # # print(win_floor, remain_budget_floor)
    # # print(win_init, remain_budget_init)
    # print("floor mean: ", np.mean(np_bid_records_floor), " floor std: ", np.std(np_bid_records_floor))
    # print("init mean: ", np.mean(np_bid_records_init), " floor std: ", np.std(np_bid_records_init))
    #
    # t_index = list(range(len(bid_records_floor)))
    # cpi_floor = cpi(win_floor, B, remain_budget_floor)
    # cpi_init = cpi(win_init, B, remain_budget_init)
    # axe[2, 1].set_xlabel('Auction time step')
    # # axe[1, 1].set_ylabel('Bid price')
    # axe[2, 1].plot(t_index, bid_records_floor, label="FloorMode_wins_" + str(win_floor) + '_cpi_' + str(cpi_floor))
    # axe[2, 1].plot(t_index, bid_records_init, label='InitMode_wins_' + str(win_init) + '_cpi_' + str(cpi_init))
    # axe[2, 1].legend()
    # axe[2, 1].set_title('binit_10.0')
    #
    #
    #
    # fig.tight_layout()
    # plt.subplots_adjust(top=0.99, bottom=0.01, hspace=1., wspace=0.)
    # fig.subplots_adjust(top=0.85)
    # fig.suptitle('B_50_bfloor_0.1_bcap_10_stepsz_0.05')
    #
    # plt.show()



