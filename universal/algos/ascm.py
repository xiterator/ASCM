import math
from universal.result import ListResult
from universal.algo import Algo
import numpy as np
import pandas as pd
import cvxpy as cp
from universal import tools


class ASCM(Algo):
    PRICE_TYPE = 'absolute'

    def __init__(self, window=5, rho_l_alpha=0.15, rho_h_alpha=0.45, rho_l_beta=0.654, rho_h_beta=0.734, subsets_comb=31,
                         lam=0.002, w_alpha=9, w_beta=5):
        super(ASCM, self).__init__(min_history=window)
        self.window = window
        self.rho_l_alpha = rho_l_alpha
        self.rho_h_alpha = rho_h_alpha
        self.rho_l_beta = rho_l_beta
        self.rho_h_beta = rho_h_beta
        self.w_beta = w_beta
        self.w_alpha = w_alpha
        self.histLen = 0
        self.subsets_comb = subsets_comb
        self.lam = lam

    def init_weights(self, m):
        return np.ones(m) / m

    def variance(self, data):
        """
            calculate the variance

            :param data: asset price data of history (5 days)
            :return: variance result
        """
        m = sum(data) / len(data)
        # var_result = sum((xi - m) ** 2 for xi in data) / len(data)
        var_result = sum((abs(xi - m)) ** 2 for xi in data)
        return var_result

    def avgPrice_set(self, history, low_limit_ratio, high_limit_ratio, avg_window):
        """
            Divide subsets based on average price

            :param history:
            :param low_limit_ratio: back percent
            :param high_limit_ratio: top percent
            :param avg_window: calculate the average price according to the data in avg_window
            :return:    set_a: union of set_ha and set_la
                        set_a_c: the complement of set_a
                        set_ha: the index set composed of asset indices with asset average prices higher than or equal
                                to the stock_num * low_limit_ratio-th asset variance in the sorted assets in ascending
                                order of average price
                        set_la: the index set composed of asset indices with asset average prices less than or equal to
                                the stock_num * high_limit_ratio-th asset variance in the sorted assets in ascending
                                order of average prices
        """
        stock_num = history.shape[1]
        set_a = [0 for i in range(stock_num)]
        set_a_c = [1 for i in range(stock_num)]
        set_ha = [0 for i in range(stock_num)]
        set_la = [0 for i in range(stock_num)]
        avgPrice_res = []
        for i in range(stock_num):
            if history.shape[0] < avg_window:
                avgPrice_res.append(sum(history.iloc[:, i]) / history.shape[0])
            else:
                avgPrice_res.append(sum(history.iloc[-avg_window:, i]) / avg_window)
        price_rank = list(np.array(avgPrice_res).argsort())

        for i in range(stock_num):
            if i <= stock_num * low_limit_ratio or i >= stock_num * high_limit_ratio:

                set_a[price_rank[i]] = 1
                set_a_c[price_rank[i]] = 0
                if i <= stock_num * low_limit_ratio:
                    set_la[price_rank[i]] = 1
                else:
                    set_ha[price_rank[i]] = 1
        return set_a, set_a_c, set_ha, set_la

    def varience_set(self, history, low_limit_ratio, high_limit_ratio, var_window):
        """
            Divide subsets based on variance

            :param history:
            :param low_limit_ratio: back percent
            :param high_limit_ratio: top percent
            :param var_window: calculate the variance according to the data in var_window
            :return:    set_v: union of set_hv and set_lv
                        set_v_c: the complement of set_v
                        set_hv: the index set composed of asset indices with asset variances higher than or equal to the
                                stock_num * high_limit_ratio-th asset variance in the sorted assets in ascending order
                                of variance
                        set_lv: the index set composed of asset indices with asset variances less than or equal to the
                                stock_num * low_limit_ratio-th asset variance in the sorted assets in ascending order
                                of variance
        """

        # calculate all assets variance and sort by result
        var_res = []
        stock_num = history.shape[1]
        set_v = [0 for i in range(stock_num)]
        set_v_c = [1 for i in range(stock_num)]
        set_hv = [0 for i in range(stock_num)]
        set_lv = [0 for i in range(stock_num)]
        for i in range(stock_num):
            var_res.append(self.variance(history.iloc[-var_window:, i]))
        var_rank = list(np.array(var_res).argsort())

        for i in range(stock_num):
            if i < int(stock_num * low_limit_ratio) or i >= int(stock_num * high_limit_ratio):
                set_v[var_rank[i]] = 1
                if i < int(stock_num * low_limit_ratio):
                    set_lv[var_rank[i]] = 1
                else:
                    set_hv[var_rank[i]] = 1
                set_v_c[var_rank[i]] = 0
        return set_v, set_v_c, set_hv, set_lv

    def calProjectVector(self, vector, subset):
        '''
            Calculate the projection vector of a vector on one subset

            :param vector:
            :param subset: list of selected asset indies
            :return: a projection vector
        '''
        ProjectVector = cp.Variable((len(vector)))
        constraints = [ProjectVector.T @ subset == 1.0, sum(ProjectVector) == 1, ProjectVector >= 0,
                       ]
        objective = cp.Minimize(cp.norm(vector - ProjectVector, 2))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CVXOPT)
        return ProjectVector.value

    def calAllProjectVectors(self, vector, sets):
        '''
            Calculate the different projection vectors of a vector on all subsets

            :param vector:
            :param sets: list of different asset subsets
            :return: list of projection vector
        '''
        projectVectors = []
        for i in range(sets.shape[0]):
            projV = self.calProjectVector(vector, sets[i])
            projectVectors.append(projV)
        return projectVectors

    def calDistance(self, vector1, vector2):
        '''
            Calculate the Euclidean distance between the vector1 and vector2

            :param vector1:
            :param vector2:
            :return: the Euclidean distance
        '''
        if len(vector1) != len(vector2):
            return -1
        distance = 0
        for i in range(len(vector1)):
            distance += (vector1[i] - vector2[i]) ** 2
        return math.sqrt(distance)

    def oneVectorNorm2(self, vector):
        '''
            Calculate the 2-norm of vector
            :param vector:
            :return:
        '''
        res = 0
        for i in range(len(vector)):
            res += vector[i] ** 2
        return math.sqrt(res)

    def findBestSubset(self, b, delta, sets, proj_b):
        '''
        :param b: balance
        :param delta: the price-trend vector
        :param sets: list of different subsets
        :param proj_b: list of projection balance
        :return:    i_k:the index of the subset that causes the maximum loss
                    distances[i_k]: the Euclidean distance between b and the proj_b in the i_k-th subset
                    loss[i_k]: the loss value in the i_k-th subset

        '''
        distances = []
        loss = []
        for i in range(sets.shape[0]):
            distance = self.calDistance(b, proj_b[i])
            expected_return = np.dot(b, delta)
            distances.append(distance)
            loss.append(self.lam * distance - expected_return)
        i_k=loss.index(max(loss))
        return i_k, distances[i_k], loss[i_k]


    def oneIteration(self, b, sets, delta):
        '''
            One iteration process

            :param b:balance
            :param sets:list of subsets
            :param delta:price-trend vector
            :return: the updated balance
        '''
        projBalance = self.calAllProjectVectors(b, sets)
        i, distance, loss = self.findBestSubset(b, delta, sets, projBalance)

        # calculate subgradient
        g_k = -delta + self.lam * (b - projBalance[i]) / distance

        g_k_norm = self.oneVectorNorm2(g_k)
        delta_norm = self.oneVectorNorm2(delta)

        # calculate afa
        if max(g_k) != 0 or min(g_k) != 0:
            afa = (loss + delta_norm ** 2) / g_k_norm ** 2
        else:
            afa = 1

        # update balance
        new_b = b - afa * g_k
        return new_b

    def unionSet(self, set1, set2):
        '''
            Obtain the union of set1 and set2

            :param set1:
            :param set2:
            :return: set is the union of set1 and set2
        '''

        set = []
        for i in range(len(set1)):
            if set1[i] == 1 or set2[i] == 1:
                set.append(1)
            else:
                set.append(0)
        return set

    def intersecSet(self, history, set1, set2):
        '''
            Obtain the intersection of set1 and set2

            :param history:
            :param set1:
            :param set2:
            :return: set is the intersection of set1 and set2
        '''

        set = []
        count = 0
        for i in range(len(set1)):
            if set1[i] == 1 and set2[i] == 1:
                set.append(1)
                count = count + 1
            else:
                set.append(0)
        if count == 0: 
            var_res = []
            stock_num = history.shape[1]
            for i in range(stock_num):
                var_res.append(self.variance(history.iloc[-self.w_alpha:, i]))
            var_rank = list(np.array(var_res).argsort())
            set[var_rank[-1]] = 1
        return set

    def getSets(self, history, set1, set2):
        '''
            Obtain the intersection of set1 and set2 and the list composed of these two sets
            :param set1:
            :param set2:
            :return:
        '''
        intersectionSet = self.intersecSet(history, set1, set2)
        sets = np.array([set1, set2])
        return intersectionSet, sets

    def step(self, x, last_b, history):
        """
            :param x: the last row data of history
            :param last_b:
            :param history:
            :return:
        """
        b = [0 for i in range(history.shape[1])]
        price_trend_vector = [0 for i in range(history.shape[1])]
        self.histLen = history.shape[0]
        stock_num = history.shape[1]

        # construct various subsets
        set_v, set_v_c, set_v_h, set_v_l = self.varience_set(history, self.rho_l_alpha, self.rho_h_alpha, self.w_alpha)
        set_a, set_a_c, set_a_h, set_a_l = self.avgPrice_set(history, self.rho_l_beta, self.rho_h_beta, self.w_beta)
        u = [1 for i in range(history.shape[1])]

        s_hv = set_v_h
        s_mv = set_v_c
        s_lv = set_v_l
        s_ha = set_a_h
        s_ma = set_a_c
        s_la = set_a_l
        s_hlv = self.unionSet(s_hv, s_lv)  # union of the s_hv and s_lv
        s_hmv = self.unionSet(s_hv, s_mv)
        s_mlv = self.unionSet(s_mv, s_lv)
        s_hla = self.unionSet(s_ha, s_la)
        s_hma = self.unionSet(s_ha, s_ma)
        s_mla = self.unionSet(s_ma, s_la)
        s_u = u  # complete set

        sets = []
        intersectionSet = []

        # Different combinations of subsets
        if self.subsets_comb == 0:
            intersectionSet, sets = self.getSets(history, s_u, s_u)
        elif self.subsets_comb == 1:
            intersectionSet, sets = self.getSets(history, s_hv, s_u)
        elif self.subsets_comb == 2:
            intersectionSet, sets = self.getSets(history, s_mv, s_u)
        elif self.subsets_comb == 3:
            intersectionSet, sets = self.getSets(history, s_lv, s_u)
        elif self.subsets_comb == 4:
            intersectionSet, sets = self.getSets(history, s_ha, s_u)
        elif self.subsets_comb == 5:
            intersectionSet, sets = self.getSets(history, s_ma, s_u)
        elif self.subsets_comb == 6:
            intersectionSet, sets = self.getSets(history, s_la, s_u)

        elif self.subsets_comb == 7:
            intersectionSet, sets = self.getSets(history, s_hlv, s_u)
        elif self.subsets_comb == 8:
            intersectionSet, sets = self.getSets(history, s_hmv, s_u)
        elif self.subsets_comb == 9:
            intersectionSet, sets = self.getSets(history, s_mlv, s_u)
        elif self.subsets_comb == 10:
            intersectionSet, sets = self.getSets(history, s_hla, s_u)
        elif self.subsets_comb == 11:
            intersectionSet, sets = self.getSets(history, s_hma, s_u)
        elif self.subsets_comb == 12:
            intersectionSet, sets = self.getSets(history, s_mla, s_u)

        elif self.subsets_comb == 13:
            intersectionSet, sets = self.getSets(history, s_hlv, s_ha)
        elif self.subsets_comb == 14:
            intersectionSet, sets = self.getSets(history, s_hlv, s_ma)
        elif self.subsets_comb == 15:
            intersectionSet, sets = self.getSets(history, s_hlv, s_la)
        elif self.subsets_comb == 16:
            intersectionSet, sets = self.getSets(history, s_hmv, s_ha)
        elif self.subsets_comb == 17:
            intersectionSet, sets = self.getSets(history, s_hmv, s_ma)
        elif self.subsets_comb == 18:
            intersectionSet, sets = self.getSets(history, s_hmv, s_la)

        elif self.subsets_comb == 19:
            intersectionSet, sets = self.getSets(history, s_mlv, s_ha)
        elif self.subsets_comb == 20:
            intersectionSet, sets = self.getSets(history, s_mlv, s_ma)
        elif self.subsets_comb == 21:
            intersectionSet, sets = self.getSets(history, s_mlv, s_la)
        elif self.subsets_comb == 22:
            intersectionSet, sets = self.getSets(history, s_hla, s_hv)
        elif self.subsets_comb == 23:
            intersectionSet, sets = self.getSets(history, s_hla, s_mv)
        elif self.subsets_comb == 24:
            intersectionSet, sets = self.getSets(history, s_hla, s_lv)

        elif self.subsets_comb == 25:
            intersectionSet, sets = self.getSets(history, s_hma, s_hv)
        elif self.subsets_comb == 26:
            intersectionSet, sets = self.getSets(history, s_hma, s_mv)
        elif self.subsets_comb == 27:
            intersectionSet, sets = self.getSets(history, s_hma, s_lv)
        elif self.subsets_comb == 28:
            intersectionSet, sets = self.getSets(history, s_mla, s_hv)
        elif self.subsets_comb == 29:
            intersectionSet, sets = self.getSets(history, s_mla, s_mv)
        elif self.subsets_comb == 30:
            intersectionSet, sets = self.getSets(history, s_mla, s_lv)

        elif self.subsets_comb == 31:
            intersectionSet, sets = self.getSets(history, s_hlv, s_hla)
        elif self.subsets_comb == 32:
            intersectionSet, sets = self.getSets(history, s_hlv, s_hma)
        elif self.subsets_comb == 33:
            intersectionSet, sets = self.getSets(history, s_hlv, s_mla)
        elif self.subsets_comb == 34:
            intersectionSet, sets = self.getSets(history, s_hmv, s_hla)
        elif self.subsets_comb == 35:
            intersectionSet, sets = self.getSets(history, s_hmv, s_hma)
        elif self.subsets_comb == 36:
            intersectionSet, sets = self.getSets(history, s_hmv, s_mla)
        elif self.subsets_comb == 37:
            intersectionSet, sets = self.getSets(history, s_mlv, s_hla)
        elif self.subsets_comb == 38:
            intersectionSet, sets = self.getSets(history, s_mlv, s_hma)
        elif self.subsets_comb == 39:
            intersectionSet, sets = self.getSets(history, s_mlv, s_mla)
        elif self.subsets_comb == 40:
            intersectionSet, sets = self.getSets(history, s_mv, s_ma)

        elif self.subsets_comb == 41:
            intersectionSet, sets = self.getSets(history, s_hv, s_ha)
        elif self.subsets_comb == 42:
            intersectionSet, sets = self.getSets(history, s_hv, s_ma)
        elif self.subsets_comb == 43:
            intersectionSet, sets = self.getSets(history, s_hv, s_la)
        elif self.subsets_comb == 44:
            intersectionSet, sets = self.getSets(history, s_mv, s_ha)
        elif self.subsets_comb == 45:
            intersectionSet, sets = self.getSets(history, s_mv, s_la)
        elif self.subsets_comb == 46:
            intersectionSet, sets = self.getSets(history, s_lv, s_ha)
        elif self.subsets_comb == 47:
            intersectionSet, sets = self.getSets(history, s_lv, s_ma)
        elif self.subsets_comb == 48:
            intersectionSet, sets = self.getSets(history, s_lv, s_la)

        # calculate price_trend_vector
        for i in range(stock_num):
            if intersectionSet[i] == 1:
                price_trend_vector[i] = self.select_price_trend(x, history, i)
        price_trend_vector_rank = np.array(price_trend_vector).argsort()

        # set initial b and old_b
        old_b = np.array([0 for i in range(stock_num)])
        b[(price_trend_vector_rank[-1])] = 1

        iterate_time = 0
        while self.calDistance(b, old_b) > 0.0005:
            old_dis = self.calDistance(b, old_b)
            old_b = b
            b = self.oneIteration(b, sets, np.array(price_trend_vector))
            b = tools.simplex_proj(b)
            iterate_time += 1
            if abs(old_dis - self.calDistance(b, old_b)) < 0.000001:
                break
            if iterate_time > 1500:
                break
        b = list(b)

        return b

    def predict(self, x, p_pred):
        """ Predict returns on next day. """
        return p_pred / x

    def select_price_trend(self, x, history, asset_id):
        '''

        :param x: the last row data of history
        :param history:
        :param asset_id: the index of one asset
        :return: the price-trend value of the asset_id-th asset
        '''

        window_data = history.iloc[-self.window:, asset_id]
        now_day = history.shape[0]

        # calculate new day's ma5,ma10 and previous day's ma5,ma10
        ma5 = sum(history.iloc[-5:, asset_id]) / 5
        ma10 = sum(history.iloc[-10:, asset_id]) / 10
        pre_ma5 = sum(history.iloc[-6:-1, asset_id]) / 5
        pre_ma10 = sum(history.iloc[-11:-1, asset_id]) / 10

        if ma5 > ma10 and pre_ma5 >= pre_ma10 and (ma5 - ma10) > (pre_ma5 - pre_ma10) and now_day > 11:
            one_return_pred = (self.predict(x[asset_id], max(window_data)))

        elif ma5 >= ma10 and pre_ma5 > pre_ma10 and (ma5 - ma10) <= (pre_ma5 - pre_ma10) and now_day > 11:
            one_return_pred = (self.predict(x[asset_id], min(window_data)))

        else:
            one_return_pred = (self.predict(x[asset_id], sum(window_data) / 5))

        return one_return_pred
