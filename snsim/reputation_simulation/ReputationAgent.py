import random
from collections import OrderedDict
import functools
from mesa import Agent
import numpy as np
import operator
from scipy.stats import truncnorm
import math
from random import shuffle
import copy
import json
import re
from itertools import product
from statistics import mean


class ReputationAgent(Agent):
    def __init__(self, unique_id, model,criminal = None, supply_list = None):

        super().__init__(unique_id, model)
        self.p = self.model.parameters  # convenience method:  its shorter
        self.wealth = 0
        self.scam_cycle_day = 0
        self.orig_scam_cycle_day = 0
        self.good = random.uniform(0,1) > self.p['chance_of_criminal'] if criminal is None else not criminal
        self.goodness = self.model.goodness_distribution.rvs() if self.good else self.model.criminal_goodness_distribution.rvs()
        self.fire_supplier_threshold = self.model.fire_supplier_threshold_distribution.rvs()
        self.scam_period = int(round(self.model.sp_distribution.rvs()))
        self.scam_inactive_period = int(round(self.model.sip_distribution.rvs()))
        self.reputation_system_threshold = self.model.reputation_system_threshold_distribution.rvs()
        self.forget_discount = self.model.forget_discount_distribution.rvs()
        self.open_to_new_experiences = self.model.open_to_new_experiences_distribution.rvs()
        self.personal_experience = OrderedDict()
        tuplist = [(good, 0) for good, chance in self.p["chance_of_supplying"].items()]
        self.days_until_shop =   OrderedDict(tuplist)
        self.criminal_days_until_shop =   OrderedDict(tuplist)
        self.shopping_pattern = OrderedDict()
        self.criminal_shopping_pattern = OrderedDict()
        self.num_criminal_consumers = OrderedDict()
        self.criminal_consumers = OrderedDict()
        self.num_criminal_consumers_product = OrderedDict()
        self.criminal_consumers_product = OrderedDict()
        self.historic_criminal_consumers = OrderedDict()
        self.historic_products = OrderedDict()
        self.hiring = OrderedDict()
        self.hiring_product = OrderedDict()
        self.product_details = {}
        self.criminal_product_details = {} #consumed products as dictionary of supplier: dictionary of category:product numbers
        self.name_increment = 0


        cumulative = 0
        self.cobb_douglas_utilities = OrderedDict()
        self.needs = []
        self.criminal_needs = []
        length = len (self.p['cobb_douglas_utilities'])
        count = 0
        self.last_last_num_product_change = {}
        self.last_num_product_change = {}
        self.never_bought_scam = True

        #every agent has its own cobb douglass utility function values for the goods,
        ## according to the distribution given in the config file

        for good, cdrv in self.model.cobb_douglas_distributions.items():
            count += 1
            rv = cdrv.rvs()
            self.cobb_douglas_utilities[good] = rv  if cumulative + rv < 1 and count < length else (1-cumulative if 1-cumulative > 0 else 0)
            cumulative = cumulative + rv

        self.cobb_douglas_utilities_original = copy.deepcopy(self.cobb_douglas_utilities)
        self.supplying = OrderedDict()
        supplying_chance_dist = self.p["chance_of_supplying"] if self.good else self.p["criminal_chance_of_supplying"]
        if supply_list is None:
            supply_list = []
            for good, chance in supplying_chance_dist.items():
                if random.uniform(0,1) < chance:
                    #price_fract = self.model.cobb_douglas_distributions[good].rvs()
                    #price = self.p['min_price']+ ((self.p['max_price'] - self.p['min_price'])*price_fract)
                    supply_list.append(good)
                    price = self.model.price_distributions[good].rvs() if self.good else self.model.criminal_price_distributions[good].rvs()
                    self.supplying[good]= price
                    self.model.suppliers[good].add(unique_id)
                    self.model.initialize_rank(unique_id)
                    self.model.reset_current_rank(unique_id)
                    if not self.good:
                        self.model.criminal_suppliers[good].add(self.unique_id)
        else:
            for good in supply_list:
                #price_fract = self.model.cobb_douglas_distributions[good].rvs()
                #price = self.p['min_price']+ ((self.p['max_price'] - self.p['min_price'])*price_fract)
                price = self.model.price_distributions[good].rvs() if self.good else self.model.criminal_price_distributions[good].rvs()
                self.supplying[good]= price
                self.model.suppliers[good].add(unique_id)
                self.model.initialize_rank(unique_id)
                self.model.reset_current_rank(unique_id)
                if not self.p['product_mode']:
                    self.initialize_criminal_ring(good)



        tuplist = [(good,[]) for good, chance in self.p["chance_of_supplying"].items()]
        self.suppliers = OrderedDict(tuplist)

        self.historic_products = {good: set() for good, _ in model.suppliers.items()}

        for good, needrv in self.model.need_cycle_distributions.items():
            self.shopping_pattern[good] = needrv.rvs()
            if self.p["randomize_initial_needs"]:
                self.days_until_shop[good] = random.randint(0, round(self.shopping_pattern[good]))

        good_utilities = copy.deepcopy(self.cobb_douglas_utilities_original)
        criminal_needs = set(self.p["criminal_category_list"])
        self.criminal_cobb_douglas_utilities = {
            need: info for need, info in good_utilities.items() if need in criminal_needs}
        self.criminal_update_utilities()


        tuplist = [(good, []) for good, chance in self.p["criminal_chance_of_supplying"].items()]
        self.criminal_suppliers = OrderedDict(tuplist)

        for good, needrv in self.model.criminal_need_cycle_distributions.items():
            self.criminal_shopping_pattern[good] = needrv.rvs()
            if self.p["randomize_initial_needs"]:
                self.criminal_days_until_shop[good] = random.randint(0, round(self.criminal_shopping_pattern[good]))

        if not self.good:

            self.historic_criminal_consumers = {good: set() for good, _ in model.suppliers.items()}
            if supply_list is not None and len(supply_list) > 0:
                #There will be overlap in the criminal rings that criminals go to
                if not self.p['product_mode']:
                    self.num_criminal_consumers = {good:int(round(self.model.criminal_agent_ring_size_distribution.rvs())) for good in supply_list}
                    self.criminal_consumers = {good:set() for good in supply_list}
                    self.hiring = {good: True for good in supply_list}
                self.scam_cycle_day = random.randint(0,self.scam_period -1)
                self.orig_scam_cycle_day = self.scam_cycle_day

        self.exogenous_reputation = self.get_initial_exogenous_reputation()
        self.initialize_products() if self.p['product_mode'] else {}
        self.shopping_history = []
        self.shopping_rating = {}
        self.anomaly_detection_data =[]
        self.predictiveness_data = []
        self.pending_predictiveness_data = {}
        self.t0 = None
        self.t1 = None
        self.conformity = []
        self.predictiveness = []

    def add_conformity(self,new_conformity):
        # eventually you may want to limit the time this goes back by fifo n long queue
        self.conformity.append(new_conformity)

    def avg_conformity(self):
        return mean(self.conformity)

    def add_predictiveness(self,new_predictiveness):
        # eventually you may want to limit the time this goes back by fifo n long queue
        self.predictiveness.append(new_predictiveness)

    def avg_predictiveness(self):
        return mean(self.predictiveness)

    def switch_product(self,category,product):
        #first check and see if it would make a difference in the personal name and dont change if it wouldnt,
        # but  once it makes a difference, test and decide
        #fixme: test
        #return False
        evidence = {}
        evidence['product_profit_current'] = 'product_profit_current' if self.product_profit_positive(category,product) else 'product_profit_currentNot'
        evidence['product_min_score'] = 'product_min_score' if self.product_min_score(category,product) else 'product_min_scoreNot'
        result = self.model.roll_bayes(evidence,"product_switch")
        change = False
        if result == 'product_switch':
            change = True
            self.model.add_product_switch()

        return change

    def check_and_remove_products (self):
        for category, productdict in self.products.items():
            for product, _ in productdict.items():
                if self.switch_product(category,product ):
                    self.remove_product (category,product)
                    self.new_product(category)

    def remove_product(self,category,product):
        self.criminal_consumers_product[category].pop(product)
        self.num_criminal_consumers_product[category].pop(product)
        self.hiring_product[category].pop(product)
        self.products[category].pop(product)



    def new_product(self,category):
            pid = self.model.next_product_id()
            self.products[category][pid] = {}
            self.products[category][pid] ["product_id"]= pid
            self.products[category][pid]["price"]= int(round(self.model.price_distributions[category].rvs(
                ) if self.good else self.model.criminal_price_distributions[category].rvs() ))
            self.products[category][pid]["quality"]=(self.goodness +
                                                    self.model.quality_deviation_from_supplier_distribution.rvs())
            self.products[category][pid]["production_cost"] = (self.products[category][pid]["price"] *
                                                                   self.products[category][pid]["quality"])/2
            self.products[category][pid]["black_market"] = False
            self.products[category][pid]["last_last_sold"]= 0
            self.products[category][pid]["last_last_income"]= 0
            self.products[category][pid]["last_last_cost"]= 0
            self.products[category][pid]["last_sold"]= 0
            self.products[category][pid]["last_income"]= 0
            self.products[category][pid]["last_cost"]= 0
            self.products[category][pid]["total_sold"]= 0
            self.products[category][pid]["total_income"]= 0
            self.products[category][pid]["total_cost"]= 0
            self.products[category][pid]["initialization"]= True
            self.products[category][pid]["ratings"] = {}
            self.initialize_criminal_product_ring(category,pid)

    def initialize_products(self):
        self.products = {}
        num_supplying = len(self.supplying)
        # create products for every category I am selling
        # attrs:  price, production_cost, quality, black_market, last_sold, last_cost, avg_sold, avg_cost
        for category, _ in self.supplying.items():
            num_products = int(round(self.model.num_products_supplied_distribution.rvs()
                                          /num_supplying))
            self.products[category] = {}
            for _ in range(num_products):
                pid = self.model.next_product_id()
                self.products[category][pid] = {}
                self.products[category][pid] ["product_id"]= pid
                self.products[category][pid]["price"]= int(round(self.model.price_distributions[category].rvs(
                    ) if self.good else self.model.criminal_price_distributions[category].rvs() ))
                quality = self.goodness +self.model.quality_deviation_from_supplier_distribution.rvs()
                self.products[category][pid]["quality"]= 1 if quality > 1 else 0 if quality < 0 else quality
                self.products[category][pid]["production_cost"] = (self.products[category][pid]["price"] *
                                                                       self.products[category][pid]["quality"])/2
                self.products[category][pid]["black_market"] = False
                self.products[category][pid]["last_last_sold"]= 0
                self.products[category][pid]["last_last_income"]= 0
                self.products[category][pid]["last_last_cost"]= 0
                self.products[category][pid]["last_sold"]= 0
                self.products[category][pid]["last_income"]= 0
                self.products[category][pid]["last_cost"]= 0
                self.products[category][pid]["total_sold"]= 0
                self.products[category][pid]["total_income"]= 0
                self.products[category][pid]["total_cost"]= 0
                self.products[category][pid]["initialization"]= True
                self.products[category][pid]["ratings"]= {}
                self.initialize_criminal_product_ring(category,pid)


    def reset_daily_purchase_stats(self):
        for category,cdict in self.products.items():
            for pid,pdict in cdict.items():
                pdict["last_last_sold"]= pdict["last_sold"]
                pdict["last_last_income"]= pdict["last_income"]
                pdict["last_last_cost"]= pdict["last_cost"]
                pdict["last_sold"]= 0
                pdict["last_income"]= 0
                pdict["last_cost"]= 0

        # self.shopping_history = []
        # self.shopping_rating = {}
        # self.anomaly_detection_data =[]
        # self.predictiveness_data = []
        # self.pending_predictiveness_data = {}

    def add_pending_predictiveness_data(self,t0,t1,ratings):
        day_to_add_data = self.model.daynum + int(self.p["days_until_prediction"])
        self.pending_predictiveness_data[day_to_add_data] = (t0,t1,ratings)

    def standardize_perception(self,perception):

        rating = None
        if perception >= 0:
            dd = self.p['ratings_goodness_thresholds']
            for rating_val, threshold in dd.items():
                if (perception < threshold or threshold == dd[next(reversed(dd))])and rating is None:
                    rating = rating_val

        return rating

    def map_rank_to_bayesian_number(self,rank):
        bayesian_number = 0
        rating = None
        ratings_version = self.p['ratings_goodness_thresholds']
        rtups = [(key,val * 100) for key,val in ratings_version.items()]
        ranks_version = OrderedDict(rtups)
        #for rating_val, threshold in ratings_version.items():
        for rating_val, threshold in ranks_version.items():
           if (rank < threshold or threshold == ranks_version[next(reversed(ranks_version))])and rating is None:
            #if (rank < threshold or threshold == ratings_version[next(reversed(ratings_version))])and rating is None:
                rating = rating_val
        bayesian_number = (self.p['ratings_bayesian_map'][rating]
                           if rating in self.p["ratings_bayesian_map"] else -1)
        return bayesian_number

    def add_predictive_row(self):
        if self.model.daynum in self.pending_predictiveness_data:
            t0_t1_ratings_tuple = self.pending_predictiveness_data[self.model.daynum]
            t0 = t0_t1_ratings_tuple [0]
            t1 = t0_t1_ratings_tuple [1]
            past_ratings = copy.deepcopy(t0_t1_ratings_tuple [2])
            current_ratings = {}
            for category, good_string_dict in past_ratings.items():
                if category not in current_ratings:
                    current_ratings[category] = {}
                for good_string, _ in good_string_dict.items():
                    if self.model.orig_ranks and good_string in self.model.orig_ranks:
                        current_rank = self.model.orig_ranks[good_string]
                        #current_rank_str = str(current_rank)
                        bayesian_number = self.map_rank_to_bayesian_number(current_rank)
                        current_ratings[category][good_string]= bayesian_number
            num_categories = len(self.p["chance_of_supplying"])

            new_row = []
            for i in range(num_categories):
                stri = str(i)
                val = 1 if t0[stri] else 0
                new_row.append(val)
            for i in range(num_categories):
                stri = str(i)
                val = 1 if t1[stri] else 0
                new_row.append(val)
            for category, good_string_dict in past_ratings.items():
                for good_string, val in good_string_dict.items():
                    case = copy.deepcopy(new_row)
                    if (val > 0 and category in current_ratings and good_string in current_ratings[category] and
                            current_ratings[category][good_string] > 0):
                        case.append(val)
                        case.append(current_ratings[category][good_string])
                        self.predictiveness_data.append(case)


    def add_predictive_row_all_goods(self):
        if self.model.daynum in self.pending_predictiveness_data:
            t0_t1_ratings_tuple = self.pending_predictiveness_data[self.model.daynum]
            t0 = t0_t1_ratings_tuple [0]
            t1 = t0_t1_ratings_tuple [1]
            past_ratings = copy.deepcopy(t0_t1_ratings_tuple [2])
            current_ratings = {}
            for category, good_string_dict in past_ratings.items():
                for good_string, _ in good_string_dict.items():
                    if good_string in self.model.orig_ranks:
                        current_rank = self.model.orig_ranks[good_string]
                        #current_rank_str = str(current_rank)
                        bayesian_number = self.map_rank_to_bayesian_number(current_rank)
                        current_ratings[category]= bayesian_number
            num_categories = len(self.p["chance_of_supplying"])

            dictdict = {}
            for i in range(num_categories):
                stri = str(i)
                dictdict[stri] = past_ratings[stri] if stri in past_ratings else {'empty':0}
            val_dict = {}
            for cat , adict in dictdict.items():
                val_dict.update(adict)
            keytuplelist = list(product(*dictdict.values()))
            for key_list in keytuplelist:
                new_row = []
                for i in range(num_categories):
                    stri = str(i)
                    val = 1 if t0[stri] else 0
                    new_row.append(val)
                for i in range(num_categories):
                    stri = str(i)
                    val = 1 if t1[stri] else 0
                    new_row.append(val)
                for key in key_list:
                    val = val_dict[key] if key in val_dict else 0
                    new_row.append(val)
                for i in range(num_categories):
                    stri = str(i)
                    val = current_ratings[stri] if stri in current_ratings else 0
                    new_row.append(val)
                #if any(new_row):
                self.predictiveness_data.append(new_row)


    def add_anomaly_row(self,t0,t1):
        num_categories = len(self.p["chance_of_supplying"])
        new_row = []
        for i in range(num_categories):
            stri = str(i)
            val = 1 if t0[stri] else 0
            new_row.append(val)
        for i in range(num_categories):
            stri = str(i)
            val = 1 if t1[stri] else 0
            new_row.append(val)
        #if (any(new_row)):
        self.anomaly_detection_data.append(new_row)

    def reset_shopping_stats(self):
        if self.p["rep_system_booster"] == "predictiveness":
            self.add_predictive_row()
        if len(self.shopping_history) > 1:
            self.t0 = self.shopping_history.pop(0)
            self.t1 = copy.deepcopy(self.shopping_history[0])
            #print("agent ${0} t0:${1} t1:${2}".format(self.unique_id, self.t0, self.t1))
            if self.p["rep_system"] == "anomaly":
                self.add_anomaly_row(self.t0,self.t1)
            ratings = self.shopping_rating
            if self.p["rep_system_booster"] == "predictiveness" and len(ratings)> 0:
                self.add_pending_predictiveness_data(self.t0, self.t1, ratings)
        today= {good: False for good, chance in self.p["chance_of_supplying"].items()}
        self.shopping_history.append(today)
        self.shopping_rating = {}



    def record_shopping(self,supplier_agent,supplier_id,category,pid,rating):
        good_string = "{0}.{1}".format(category, pid) if self.p['product_mode'] else category
        reputation_string = "{0}.{1}".format(supplier_id, good_string) if self.p['product_mode'] else supplier_id
        place = len(self.shopping_history) - 1
        shopdict = self.shopping_history[place]
        shopdict[category] = True
        if self.p['include_ratings'] and rating and rating != "":
            if category not in self.shopping_rating:
                self.shopping_rating[category] = {}
            bayesian_number = (self.p['ratings_bayesian_map'][rating]
                                   if rating in self.p['ratings_bayesian_map'] else 0)
            self.shopping_rating[category][reputation_string] = bayesian_number

            supplier_agent.products[category][pid]["ratings"][self.unique_id] = float(rating)



    def note_product_purchase(self,category,pid,amount,scam=False):
        if pid >=0:
            cost = 0
            if scam:
                cost = amount*(self.products[category][pid]["production_cost"]+ self.products[category][pid]["price"])
                self.products[category][pid]["last_cost"]+= cost
                self.products[category][pid]["total_cost"] += cost

                # print("supplier {0} purchased amount {1} category {2} product {3} scam {4}  cost {5}".format(
                #     self.unique_id, amount, category, pid, scam, cost))
                # if self.model.error_log:
                #     self.model.error_log.write(
                #         "supplier {0} purchased amount {1} category {2} product {3} scam {4}  cost {5}\n".format(
                #             self.unique_id, amount, category, pid, scam, cost))

            else:
                cost = amount*self.products[category][pid]["production_cost"]
                self.products[category][pid]["last_sold"]+= amount
                self.products[category][pid]["last_income"]+= amount*self.products[category][pid]["price"]
                self.products[category][pid]["last_cost"]+= cost
                self.products[category][pid]["total_sold"]+= amount
                self.products[category][pid]["total_income"] += amount*self.products[category][pid]["price"]
                self.products[category][pid]["total_cost"] += cost

    def supplier_has_available_products(self,supplier, good):
        supplier_has_available_products = True
        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        testset = self.historic_products[good] if good in self.historic_products else set()
        if supplier_agent.get_available_criminal_product(good,testset) is None:
            supplier_has_available_products = False
        return supplier_has_available_products



    def get_available_criminal_product(self,good,previously_bought):
        available = []
        if good in self.hiring_product:
            for product,hiring  in self.hiring_product[good].items():
                numprods = len(self.criminal_consumers_product[good][product])
                if (hiring and
                        not product in previously_bought and
                        numprods < self.p['max_scam_purchases'] and
                        numprods < self.num_criminal_consumers_product[good][product]):
                    available.append(product)
        random_available = random.choice(available) if len(available)> 0 else None
        return random_available

    def needs_criminal_consumer(self, supplier,good):
        needs = False
        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        if self.p['product_mode']:
            if good in supplier_agent.hiring_product:
                for product,hiring  in supplier_agent.hiring_product[good].items():
                    numproduct= len(supplier_agent.criminal_consumers_product[good][product])
                    if (hiring and
                        numproduct < self.p['max_scam_purchases'] and
                        numproduct < supplier_agent.num_criminal_consumers_product[good][product]):

                        needs = True
                    else:
                        supplier_agent.hiring_product[good][product] = False
        else:
            if (good in supplier_agent.hiring and supplier_agent.hiring[good]and
                len(supplier_agent.criminal_consumers[good]) < supplier_agent.num_criminal_consumers[good]):
                    needs = True
            else:
                supplier_agent.hiring[good] = False
        return needs

    def taken(self):
        taken = False
        if not self.good and self.p['exclusive_criminal_ring']:
            suppliers = [supplier  for good, supplierlist in self.criminal_suppliers.items()for supplier in supplierlist]
            taken = True if len(suppliers)> 0 else False
        return taken

    def adopt_criminal_supplier(self, good):
        supplier = None
        if good in self.model.criminal_suppliers:
            possible_suppliers = [supplier for supplier in self.model.criminal_suppliers[good] if (
                    self.supplier_has_available_products(supplier,good) and
                    #not (good in self.historic_products and supplier in self.historic_products[good] )and
                    supplier != self.unique_id and
                    self.needs_criminal_consumer(supplier, good))  and
                    not self.supplier_inactive(supplier) ]
            #if len(possible_suppliers) == 0:
                #print("no available products in category {0} for agent {1}".format(good, self.unique_id))
                # possible_suppliers = [supplier for supplier in self.model.criminal_suppliers[good] if (
                #     supplier != self.unique_id and
                #     self.needs_criminal_consumer(supplier, good))  and
                #     not self.supplier_inactive(supplier) ]
            if len(possible_suppliers) > 0:
                supplier = possible_suppliers[random.randint(0,len(possible_suppliers)-1)]
                supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
                if self.p['product_mode']:
                    testset = self.historic_products[good] if good in self.historic_products else set()
                    product = supplier_agent.get_available_criminal_product(good,testset)
                    self.set_criminal_supplier(supplier, good, product)
                    supplier_agent.criminal_consumers_product[good][product].add(self.unique_id)
                else:
                    self.set_criminal_supplier(supplier, good)
                    supplier_agent.criminal_consumers[good].add(self.unique_id)
                supplier_agent.historic_criminal_consumers[good].add(self.unique_id)
                #self.suppliers[good].append(supplier)

    def clear_supplierlist (self, good,supplierlist):
        if not self.good:
            for supplier in supplierlist:
                supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
                if self.p['product_mode']:
                    if good in supplier_agent.criminal_consumers_product:
                        for product,consumerset in  supplier_agent.criminal_consumers_product[good].items():
                            if self.unique_id in consumerset:
                                consumerset.remove(self.unique_id)
                else:
                    if self.unique_id in supplier_agent.criminal_consumers[good]:
                        supplier_agent.criminal_consumers[good].remove (self.unique_id)
        supplierlist.clear()

    def average_ranks(self):
        sum = 0
        for agent,rank in self.model.ranks.items():
            sum += rank
        if len(self.model.ranks)> 0:
            sum /= len(self.model.ranks)
        else:
            sum = self.reputation_system_threshold
        return sum


    def update_utility(self,good,utility):
        #  cobb douglass need actual = cobb douglas need + K(days pass need met)(cobb douglassneed)
        days_past_need_met = -self.days_until_shop[good] if self.days_until_shop[good] <= -1 else 0
        utility = (self.cobb_douglas_utilities[good]
                   + self.p['cobb_douglas_constant']*days_past_need_met*self.cobb_douglas_utilities[good])
        return utility

    def criminal_update_utility(self,good,utility):
        #  cobb douglass need actual = cobb douglas need + K(days pass need met)(cobb douglassneed)
        days_past_need_met = -self.criminal_days_until_shop[good] if self.criminal_days_until_shop[good] <= -1 else 0
        utility = (self.criminal_cobb_douglas_utilities[good]
                   + self.p['cobb_douglas_constant']*days_past_need_met*self.criminal_cobb_douglas_utilities[good])
        return utility

    def normalize_utilities(self,utilities):
        #make all the utilities add to one

        sum_utilities = sum(utilities.values())
        utilities = {k:(v/sum_utilities) for k,v in utilities.items()}
        return utilities

    def update_utilities(self):
        # cobb douglas utilities here represent the degree goods are needed.  The amount of time the good is late in
        # getting resupplied is directly proportional to its bump in the list of needed.  The most needed are pursued
        # first, which matters only when there are fewer transactions per day allowed than meets needs.

        utilities = {k:self.update_utility(k,v) for k,v in self.cobb_douglas_utilities.items()}
        self.cobb_douglas_utilities=self.normalize_utilities(utilities)


    def criminal_update_utilities(self):
        # cobb douglas utilities here represent the degree goods are needed.  The amount of time the good is late in
        # getting resupplied is directly proportional to its bump in the list of needed.  The most needed are pursued
        # first, which matters only when there are fewer transactions per day allowed than meets needs.

        utilities = {k:self.criminal_update_utility(k,v) for k,v in self.criminal_cobb_douglas_utilities.items()}
        self.criminal_cobb_douglas_utilities=self.normalize_utilities(utilities)

    def avg_profit_positive(self):
        profit = self.avg_profit()
        positive = True if profit >0 else False
        return positive

    def avg_profit(self):
        incomes = [infodict["last_income"]
                   for good, productdictdict in self.products.items()
                   for product, infodict in productdictdict.items()]
        income = sum (incomes)
        costs = [infodict["last_cost"]
                   for good, productdictdict in self.products.items()
                   for product, infodict in productdictdict.items()]
        cost = sum(costs)
        net_income = income - cost
        return net_income


    def avg_score_above_mean(self):
        avg_rank = self.model.get_current_rank(self.unique_id)
        avg_avg_rank = self.model.get_current_avg_rank()
        over = True if avg_rank > avg_avg_rank else False
        return over

    def avg_score_over_threshold(self):
        avg_rank = self.model.get_current_rank(self.unique_id)
        over = True if avg_rank > self.reputation_system_threshold else False
        return over

    def change_generations(self):
        #fixme test
        #return False
        #first check and see if it would make a difference in the personal name and dont change if it wouldnt,
        # but  once it makes a difference, test and decide
        change = False
        if not self.never_bought_scam:
            evidence = {}
            evidence['supplier_profit'] = 'supplier_profit' if self.avg_profit_positive() else 'supplier_profitNot'
            #evidence['supplier_min_score'] = 'supplier_min_score' if self.avg_score_over_threshold() else 'supplier_min_scoreNot'
            evidence['supplier_min_score'] = 'supplier_min_score' if self.avg_score_above_mean() else 'supplier_min_scoreNot'
            evidence['supplier_exogenousReputation'] = ('supplier_exogenousReputation'
                    if self.exogenous_reputation else 'supplier_exogenousReputationNot')

            result = self.model.roll_bayes(evidence,"supplier_switch")
            change = True if result == "supplier_switch" else False
        return change

    def test_and_set_generation_increment(self):
        if self.p['criminal_suppliers_change_names']:
            calc = self.calculate_present_increment(self.unique_id)
            if  calc > self.name_increment and self.change_generations():
                self.model.error_log.write("supplier {0} changed name from {1} to {2}\n".format(self.unique_id,self.name_increment,calc))
                #print("supplier {0} changed name from {1} to {2}".format(self.unique_id,self.name_increment,calc))
                self.change_name()
                self.model.add_identity_change()
            elif self.p['criminal_suppliers_replace_products']:
                self.check_and_remove_products()
                self.initialize_criminal_product_rings()
                self.reset_daily_purchase_stats()
            else:
                self.initialize_criminal_product_rings()
                self.reset_daily_purchase_stats()
        elif self.p['criminal_suppliers_replace_products']:
            self.check_and_remove_products()
            self.initialize_criminal_product_rings()
            self.reset_daily_purchase_stats()
        else:
            self.initialize_criminal_product_rings()
            self.reset_daily_purchase_stats()

    def check_and_replace_former_criminal_supplier(self,good):
        if (not self.good) and (not self.taken()):
            if good in self.criminal_suppliers:
                supplierlist = self.criminal_suppliers[good]
                for supplier in supplierlist:
                    supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
                    if (self.p['random_change_suppliers'] == 1.0
                            or self.supplier_inactive(supplier)
                            or self.p['one_review_per_product']):
                        supplierlist.remove(supplier)
                        # if self.p['product_mode']:
                        #     if good in supplier_agent.criminal_consumers_product:
                        #         for product,consumerset in supplier_agent.criminal_consumers_product[good].items():
                        #             if self.unique_id in consumerset:
                        #                 supplier_agent.criminal_consumers_product[good][product].remove(self.unique_id)
                        # else:
                        #     if self.unique_id in supplier_agent.criminal_consumers[good]:
                        #         supplier_agent.criminal_consumers[good].remove(self.unique_id)
                    elif (not self.supplier_inactive(supplier)
                            and not self.p['one_review_per_product']):
                        roll = random.uniform(0,1)
                        if roll < self.p['random_change_suppliers']:
                            if self.p['product_mode']:
                                supplierlist.remove(supplier)
                                #commented out because:
                                #the agents are left in the rings only to be taken out when the rings reset,
                                #so the suppliers wont look for agents over what it has already alotted.
                                # if good in supplier_agent.criminal_consumers_product:
                                #     for product, consumerset in supplier_agent.criminal_consumers_product[good]:
                                #         if self.unique_id in consumerset:
                                #             supplier_agent.criminal_consumers_product[good][product].remove(
                                #                 self.unique_id)
                            else:
                                supplierlist.remove(supplier)
                                # if self.unique_id in supplier_agent.criminal_consumers[good]:
                                #     supplier_agent.criminal_consumers[good].remove(self.unique_id)


                if len(supplierlist) < 1:
                    self.adopt_criminal_supplier(good)

    def incr_root_id(self, incr):
        root_id_match = self.model.id_pattern.search(self.unique_id)
        root_id = root_id_match.group(1)
        increment = root_id_match.group(2)
        new_root_id = int(root_id) +incr
        new_id = str(new_root_id) + "-" + increment
        return new_id

    def alias_with_increment(self, alias_int, id_str):
        #given an integer alias , get the increment off of the name
        #put it on the alias and return it.  its a possible past alias
        #that shold be removed

        root_id_match = self.model.id_pattern.search(id_str)
        increment = root_id_match.group(2)
        new_id = str(alias_int) + "-" + increment
        return new_id

    def new_id(self):
        #use the original id value, as an agent may have changed ids
        original_unique_id = self.model.orig[self.unique_id]
        root_id_match = self.model.id_pattern.search(original_unique_id)
        root_id = root_id_match.group(1)
        increment = root_id_match.group(2)
        new_increment = int(increment) +1
        new_id = root_id + "-" + str(new_increment)
        return new_id

    def leave_and_enter(self):
        # an agent dies in that its memory is deleted,
        # and another agent with his supplying and
        # criminality requirements takes its place.


        for good, price in self.supplying.items():
            if self.unique_id in self.model.suppliers[good]:
                self.model.suppliers[good].remove(self.unique_id)
                if not self.good and self.unique_id in self.model.criminal_suppliers[good]:
                    self.model.criminal_suppliers[good].remove(self.unique_id)

        self.model.finalize_rank(self.unique_id)
        self.model.average_rank_history.flush()
        new_id = self.new_id()
        self.model.orig[new_id] = new_id
        self.model.m[new_id] = self.model.m[self.unique_id]
        supply_list = [item for item, _ in self.supplying.items()]
        self.__init__(new_id, self.model, (not self.good), supply_list)

    def step(self):
        #print ("first line of ReputationAgent step")
        #first increment the  number of days that have taken place to shop for all goods
        #tempDict = {good: days-1 for good, days in self.days_until_shop.items() if days > 0 }

        if len(self.supplying) <= 0:
            roll = random.uniform(0, 1) if self.p["consumer_chance_of_leaving_and_entering"]> 0.0 else 1.0
            if roll < self.p["consumer_chance_of_leaving_and_entering"]:
                self.leave_and_enter()
        else:
            roll = random.uniform(0, 1) if self.p["supplier_chance_of_leaving_and_entering"]> 0.0 else 1.0
            if roll < self.p["supplier_chance_of_leaving_and_entering"]:
                self.leave_and_enter()

        self.reset_shopping_stats()
        tempDict = {good: days-1 for good, days in self.days_until_shop.items()  }
        self.days_until_shop.update(tempDict)
        tempDict = {good: days-1 for good, days in self.criminal_days_until_shop.items()  }
        self.criminal_days_until_shop.update(tempDict)


        if (self.p['suppliers_are_consumers'] or len(self.supplying)>= 1):
            self.scam_cycle_day += 1
            if (self.scam_cycle_day % self.scam_period)==0:
                self.scam_cycle_day = 0
        # kick out suppliers. See how many trades you will make.
        # initialize todays goods
        # go through and make a list of needs in order

        if self.model.daynum > 0 and len(self.supplying)>= 1 and not self.good:
            self.test_and_set_generation_increment()

        if (self.p['suppliers_are_consumers'] or len(self.supplying)< 1):

            #have agents start out with minute, non zero supplies of all goods, to make sure cobb douglas works#fixme should bad agent good be gien to bad agents
            #tuplist = [(good, random.uniform(0.1,0.2)) for good, chance in self.p["chance_of_supplying"].items()]
            #self.goods = OrderedDict(tuplist)
            num_trades =int(round(self.model.transactions_per_day_distribution.rvs()))
            if num_trades == 0:
                num_trades = 1  #If there is less than one trade per day, the needs cycle determine whether the trade is made


            #we offer a more efficient version of cobb_douglas, which is a needs draw

            # utilities = OrderedDict()
            # for testgood, u in self.cobb_douglas_utilities.items():
            #     cumulative = 1
            #     for good, utility in self.cobb_douglas_utilities.items():
            #         goodnum = self.goods[good]+1 if testgood == good else self.goods[good]
            #         cumulative = cumulative * pow(goodnum,utility)
            #     utilities[testgood]= cumulative

            #self.needs = sorted(utilities.items(), key=operator.itemgetter(1), reverse=True)
            self.update_utilities()
            p = [v for v in list(self.cobb_douglas_utilities.values()) if v > 0]
            n = len(p)
            keys = list(self.cobb_douglas_utilities.keys())[0:n]
            wants = list(np.random.choice(keys, n, p=p, replace=False))
            self.needs = [want for want in wants if
                          self.days_until_shop[want] < 1]  # todo: sorting is faster and almost the same
            if not self.p['average_reputation_system_threshold'] and self.p['average_reputation_system_threshold']:
                self.reputation_system_threshold = self.average_ranks()
            self.multiplier = 1
            if num_trades and  num_trades < len(self.needs):
                self.needs = self.needs[:num_trades]
            elif self.needs:
                self.multiplier = int(round(num_trades/len (self.needs)))

            if not self.good:
                #tuplist = [(good, random.uniform(0.1, 0.2)) for good, chance in self.p["criminal_chance_of_supplying"].items()]
                #self.criminal_goods = OrderedDict(tuplist)
                num_criminal_trades = int(round(self.model.criminal_transactions_per_day_distribution.rvs()))
                if num_criminal_trades == 0:
                    num_criminal_trades = 1  # If there is less than one trade per day, the needs cycle determine whether the trade is made

                self.criminal_update_utilities()
                p = [v for v in list(self.criminal_cobb_douglas_utilities.values()) if v > 0]
                n = len(p)
                keys = list(self.criminal_cobb_douglas_utilities.keys())[0:n]
                wants = list(np.random.choice(keys, n, p=p, replace=False))
                self.criminal_needs = [want for want in wants if self.criminal_days_until_shop[
                    want] < 1]  # todo: sorting is faster and almost the same
                self.criminal_multiplier = 1
                if num_criminal_trades and  num_criminal_trades < len(self.criminal_needs):
                    self.criminal_needs = self.criminal_needs[:num_criminal_trades]
                elif self.criminal_needs:
                    self.criminal_multiplier = int(round(num_criminal_trades/len (self.criminal_needs)))


    def consumer_to_supplier_avg_rating(self,supplier):
        sum = 0
        count = 0
        for category,supplierdict in self.personal_experience.items():
            if supplier in supplierdict:
                sum += supplierdict[supplier][1]
                count += supplierdict[supplier][0]
        perception = sum/count if count > 0 else -1

        rating = self.standardize_perception(perception)

        return rating



    def update_personal_experience(self, good, supplier, rating):
        #rating = float(rating_str)
        if not good in self.personal_experience:
            self.personal_experience[good]= OrderedDict()
        if supplier not in self.personal_experience[good] :
            self.personal_experience[good][supplier] = (1,rating)
        else:
            times_rated, past_rating = self.personal_experience[good][supplier]
            new_times_rated = times_rated + 1
            new_rating = rating
            if rating > max(self.p['fire_supplier_threshold'][0] , past_rating): #if the rating is bad, mark him despite the past
                #remember forget_discount of the past, and add that onto your memory of the present
                forgotton = (1-self.forget_discount) * times_rated
                new_rating = ((times_rated - forgotton) * past_rating + (1+ forgotton)* rating)/new_times_rated
            self.personal_experience[good][supplier] = (new_times_rated, new_rating)

    def choose_with_threshold(self,good, under_threshold):
        #only good agents come here, and when there is a reputation system.

        # sort takes a long time but we do it if choice method is winner take all
        #choose from your own experiences by your past favorite, one of your past favorites,
        # or using the reputation system.  todo:  add more choice methods that include combinations of these
        winner = None
        if self.model.ranks:
            if self.p['choice_method'] == "thresholded_random":

                # thresholded random, using the reputation system AND past experience

                over_threshold = [agent for agent, rating in self.model.ranks.items()
                                  if
                                  rating > self.reputation_system_threshold
                                  and int(agent) in self.model.suppliers[good]
                                  #and good in self.model.agents[self.model.orig[int(agent)]].supplying
                                  and int(agent) not in under_threshold
                                  and not self.supplier_inactive(int(agent))]
                if len(over_threshold)> 0:
                    roll = np.random.randint(0, len(over_threshold))
                    winner = int(over_threshold[roll])
            elif self.p['choice_method'] == "winner_take_all":

                non_criminal_experiences = {agent_string: rating for agent_string, rating in self.model.ranks.items()
                                            if
                                            (rating > self.reputation_system_threshold)
                                            and self.parse(agent_string)['agent'] in self.model.suppliers[good]
                                            # and (good in self.model.agents[self.model.orig[int(agent)]].supplying )
                                            and (not self.parse(agent_string)['agent'] in under_threshold)
                                            and (not self.supplier_inactive(self.parse(agent_string)['agent']))}
                sorted_suppliers = sorted(non_criminal_experiences.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_suppliers) > 0:
                    winner = sorted_suppliers[0][0]

            elif self.p['choice_method'] == "roulette_wheel":

                    non_criminal_experiences = {agent_string: rating for agent_string, rating in self.model.ranks.items()
                                                if
                                                (rating > self.reputation_system_threshold)
                                                and self.parse(agent_string)['agent'] in self.model.suppliers[good]
                                                #and (good in self.model.agents[self.model.orig[int(agent)]].supplying )
                                                and (not self.parse(agent_string)['agent'] in under_threshold)
                                                and (not self.supplier_inactive(self.parse(agent_string)['agent']))}
                    ratings_sum = sum([rating for key, rating in non_criminal_experiences.items()])
                    if ratings_sum > 0:
                        roll = random.uniform(0, ratings_sum)
                        cumul = 0
                        for key, rating in non_criminal_experiences.items():
                            if winner is None:
                                cumul = cumul + rating
                                if cumul > roll:
                                    winner = key

        return winner

    def parse(self,agent_string):
        parse = {}
        agent_string_split = agent_string.split(".")
        parse['agent']= agent_string_split[0]  #if self.p['product_mode'] else int(agent_string)

        parse['category']= agent_string_split[1]if len(agent_string_split)==3 else None
        parse['product']= int(agent_string_split[2])if len(agent_string_split)==3 else None
        return parse



    def supplier_inactive(self, supplier):
        #inactive if agent is in SAP or if agents orig is in sap, or if the agent is hiding its name.
        #now, inactive means just inactive to bad agents.  The supplier is always active to good agents
        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        if supplier_agent.unique_id != supplier:
            inactive = True
        elif self.good:
            inactive = False
        else:
            inactive = True

            if supplier_agent.good:
                inactive = False
            else:
                generation_increment = self.get_criminal_suppliers_increment(supplier)
                #if (supplier_agent.scam_cycle_day >= supplier.scam_inactive_period):
                scam_active_period = supplier_agent.scam_period - supplier_agent.scam_inactive_period
                if (supplier_agent.scam_cycle_day < scam_active_period):
                    inactive = False
                if (self.model.int_id(supplier) != self.model.int_id(self.model.orig[supplier])+generation_increment) :
                #if (supplier != self.model.orig[supplier]+generation_increment) :
                    inactive = True

        return inactive


    def product_profit_positive(self,good,product):

        profit = True if self.products[good][product]["last_income"]-self.products[good][product]["last_cost"] > 0 else False
        return profit

    def product_best_score(self,good,product):
        rankstring = '{0}.{1}.{2}'.format(self.unique_id,good,product)
        rank = self.model.ranks[rankstring] if rankstring in self.model.ranks else -1
        score = True if rank > self.p['ratings_goodness_thresholds'][self.best_rating()] else False
        return score

    def product_min_score(self,good,product):
        rankstring = '{0}.{1}.{2}'.format(self.unique_id,good,product)
        rank = self.model.ranks[rankstring] if self.model.ranks and rankstring in self.model.ranks else -1
        score = True if rank > self.reputation_system_threshold else False
        return score

    def product_initial_decisions(self,good,product):
        if good not in self.last_num_product_change:
            self.last_num_product_change[good] = {}
        if good not in self.last_last_num_product_change:
            self.last_last_num_product_change[good] = {}
        if product not in self.last_num_product_change[good]:
            self.last_num_product_change[good][product] = None
        if product not in self.last_last_num_product_change[good]:
            self.last_last_num_product_change[good][product] = None

        is_initial = True if (self.last_last_num_product_change[good][product] is None
                     or self.last_num_product_change[good][product]  is None) else False
        return is_initial


    def product_Dprofit(self,category,product):
        #     "product_Dprofit":
        #         "product_DprofitLoss"
        #         "product_DprofitGain"
        #         "product_DprofitEven"

        product_Dprofit = "product_DprofitEven"
        profit = self.products[category][product]["last_income"]-self.products[category][product]["last_cost"]
        former_profit = self.products[category][product]["last_last_income"]-self.products[category][product]["last_last_cost"]
        if profit - former_profit > self.p["profit_equality_tolerance"]:
            product_Dprofit = "product_DprofitGain"
        elif former_profit - profit > self.p["profit_equality_tolerance"]:
            product_Dprofit = "product_DprofitLoss"

        return product_Dprofit

    def product_reputation(self,category,product):
        #"product_reputation":
        #         "product_reputation_belowAverage"
        #         "product_reputation_medium"
        #         "product_reputation_topTen"

        reputation = "product_reputation_medium"

        if category in self.model.topNPercent and  product in self.model.topNPercent[category]:
            reputation = "product_reputation_topTen"
        elif category in self.model.bottomNPercent and   product in self.model.bottomNPercent[category]:
            reputation = "product_reputation_belowAverage"

        return reputation

    def num_bought_scam_change(self,category, product):

        # "num_bought_scams_initial": looking
        # for ""product_initialScam_higher",", "product_initialScam_lower", "product_initialScam_even", in "product_initialScam"
        # "num_bought_scams": looking
        # for "product_scam_higher", "product_scam_lower", "product_scam_even", in "product_scam"
        # RV's:
        # "num_bought_scams_initial":
        #     "product_reputation":
        #         "product_reputation_belowAverage"
        #         "product_reputation_medium"
        #         "product_reputation_topTen"
        #     "supplier_exogenousReputation":
        #         "supplier_exogenousReputation"
        #         "supplier_exogenousReputationNot"
        #     "product_profit_current":
        #         "product_profit_current"
        #         "product_profit_currentNot"
        #
        # "num_bought_scams":
        #     "product_Dprofit":
        #         "product_DprofitLoss"
        #         "product_DprofitGain"
        #         "product_DprofitEven"
        #     "product_profit_current":
        #         "product_profit_current"
        #         "product_profit_currentNot"
        #     "supplier_lastScam":
        #         "supplier_lastScam_higher"
        #         "supplier_lastScam_lower"
        #     "supplier_lastLastScam":
        #         "supplier_lastLastScam_higher"
        #         "supplier_lastLastScam_lower"

        evidence = {}
        winner = "product_scam_even"
        if self.product_initial_decisions(category,product):
            evidence['product_reputation'] = self.product_reputation(category,product)
            evidence['product_profit_current'] = 'product_profit_current' if self.product_profit_positive(category, product) else 'product_profit_currentNot'
            evidence['supplier_exogenousReputation'] = 'supplier_exogenousReputation' if self.exogenous_reputation else 'supplier_exogenousReputationNot'
            winner = self.model.roll_bayes(evidence, "num_bought_scams_initial")
        else:
            evidence['product_Dprofit'] = self.product_Dprofit(category,product)
            evidence['product_profit_current'] = 'product_profit_current' if self.product_profit_positive(category, product) else 'product_profit_currentNot'
            evidence['supplier_lastScam'] = 'supplier_lastScam_higher' if (
                self.last_num_product_change[category][product] == "num_bought_scams_higher")else 'supplier_lastScam_lower'
            evidence['supplier_lastLastScam'] = 'supplier_lastLastScam_higher' if (
                    self.last_last_num_product_change[category][product] == "num_bought_scams_higher") else 'supplier_lastLastScam_lower'
            winner = self.model.roll_bayes(evidence, "num_bought_scams")
        if winner != "product_scam_even" and winner != "product_initialScam_even":
            self.last_last_num_product_change [category][product]= self.last_num_product_change[category][product]
            self.last_num_product_change[category][product] = winner
        return winner

    def initialize_criminal_product_ring(self, category, product):
        if self.p['criminal_suppliers_purchase_reviewers']:

            if not category in self.hiring_product:
                self.hiring_product[category]= {}
            self.hiring_product[category][product] = True

            if not category in self.criminal_consumers_product:
                self.criminal_consumers_product[category]= {}
            self.criminal_consumers_product[category][product] = set()

            if self.products[category][product]['initialization']:
                self.products[category][product]['initialization']=False
                macroview = self.model.config['parameters']['macro_view']
                if macroview and self.model.config['macro_views'][macroview]['supplement'][
                         "overwrite_criminal_agent_ring_size"] == "market_research":
                    if not category in self.num_criminal_consumers_product:
                        self.num_criminal_consumers_product[category]= {}
                    if self.exogenous_reputation:
                        self.num_criminal_consumers_product[category][product] = 0
                    else:
                        purch = self.model.config['macro_views'][macroview]['supplement']['purch']
                        profit_margin = self.model.config['macro_views'][macroview]['supplement']['profit_margin']
                        beta = self.model.config['macro_views'][macroview]['supplement']['beta']
                        alpha = self.model.config['macro_views'][macroview]['supplement']['alpha']

                        #for category,price in self.supplying.items():
                        #for product in self.product_details[category]:
                        price = self.products[category][product]['price']
                        self.num_criminal_consumers_product[category][product] = 0 if price <= 0 else (
                            (purch/math.pow(price,beta))*
                            ((price*profit_margin)/(price + pow(price,alpha)))
                            )
                else:
                    self.num_criminal_consumers_product[category][product] = int(round(self.model.criminal_agent_ring_size_distribution.rvs()))
            else:
                winner = self.num_bought_scam_change(category,product)

                change = self.p["change_num_scams_incr"] if winner == 'product_scam_higher' else (
                    -self.p["change_num_scams_incr"] if winner == 'product_scam_lower' else (0))
                self.num_criminal_consumers_product[category][product] += change
                if self.num_criminal_consumers_product[category][product] < self.p['min_scam_purchases']:
                    self.num_criminal_consumers_product[category][product] = self.p['min_scam_purchases']
                elif self.num_criminal_consumers_product[category][product] > self.p['max_scam_purchases']:
                    self.num_criminal_consumers_product[category][product] = self.p['max_scam_purchases']
                #if self.num_criminal_consumers_product[category][product] > 0 :
                if self.model.error_log:
                    self.model.error_log.write("supplier {4} product {0} category {1} num to buy reviews changes {2} to {3}\n".format(
                        product,category,change,self.num_criminal_consumers_product[category][product],self.unique_id))

                #print("supplier {4} product {0} category {1} num to buy reviews changes {2} to {3}".format(
                 #   product,category,change,self.num_criminal_consumers_product[category][product],self.unique_id))
        else:
            self.num_criminal_consumers_product[category][product] = 0

    def initialize_criminal_ring(self, category):
        self.hiring[category]= True
        macroview = self.model.config['parameters']['macro_view']
        if macroview and self.model.config['macro_views'][macroview]['supplement'][
                 "overwrite_criminal_agent_ring_size"] == "market_research":
            purch = self.model.config['macro_views'][macroview]['supplement']['purch']
            profit_margin = self.model.config['macro_views'][macroview]['supplement']['profit_margin']
            beta = self.model.config['macro_views'][macroview]['supplement']['beta']
            alpha = self.model.config['macro_views'][macroview]['supplement']['alpha']
            #error why categorys done multiple times??
            #for category,price in self.supplying.items():
            price = self.supplying[category]

            self.num_criminal_consumers[category] = 0 if price <= 0 else (
                (purch/math.pow(price,beta))*
                ((price*profit_margin)/(price + pow(price,alpha)))
                )

    def clear_products(self):

        for category,_ in self.supplying.items():
            self.criminal_consumers_product[category]={}
            self.num_criminal_consumers_product[category]={}
            self.hiring_product[category]= {}
            self.products[category] = {}

    def initialize_criminal_product_rings(self):
        for category, pdict in self.products.items():
            for product, _ in pdict.items():
                self.initialize_criminal_product_ring(category, product)


    def change_name(self):
        #move stuff here to find out what name to change to

        calc = self.calculate_present_increment(self.unique_id)
        self.name_increment = calc
        supplier = self.incr_root_id(self.get_criminal_suppliers_increment(self.unique_id))
        self.model.orig[supplier] = self.unique_id
        if not self.good:
            if self.p['product_mode']:
                if self.p["change_product_names_when_change_supplers_name"]:
                    self.clear_products()
                    self.initialize_products()
                else:
                    self.initialize_criminal_product_rings()
                    self.reset_daily_purchase_stats()
            else:
                self.initialize_criminal_ring()

            for good, price in self.supplying.items():
                self.model.criminal_suppliers[good].add(supplier)
                self.model.suppliers[good].add(supplier)
                self.model.initialize_rank(supplier)
                self.model.reset_current_rank(supplier)
                scam_periods_so_far = (self.model.daynum // self.scam_period) + 1
                for i in range(scam_periods_so_far):
                    alias = (i * self.p['num_users']) + self.model.int_id(self.model.orig[supplier])
                    alias_str = self.alias_with_increment(alias,supplier)
                    if alias_str != supplier and alias_str in self.model.suppliers[good]:
                        self.model.suppliers[good].remove(alias_str)
                        if not self.good and alias_str in self.model.criminal_suppliers[good]:
                            self.model.criminal_suppliers[good].remove(alias_str)

            for i in range(scam_periods_so_far):
                alias = (i * self.p['num_users']) + self.model.int_id(self.model.orig[supplier])
                alias_str = self.alias_with_increment(alias, supplier)
                self.model.finalize_rank(alias_str)
                self.model.average_rank_history.flush()

    def choose_partners(self):
        #one good per each time you enter.  change this by looping on goods per day if you have more goods that
        #times that choose_partners is called.

        if self.needs:
            good = self.needs.pop(False)
            for i in range(self.multiplier):

                for supplier in self.suppliers[good]:
                    if self.p['random_change_suppliers'] == 1.0 or self.supplier_inactive(supplier):
                                self.suppliers[good].remove(supplier)
                    elif (good in self.personal_experience and supplier in self.personal_experience[good]
                        and len(self.personal_experience[good][supplier]) > 0):
                        if self.personal_experience[good][supplier][1] < self.fire_supplier_threshold:
                            self.suppliers[good].remove(supplier)
                        else:
                            roll = random.uniform(0, 1) if random.uniform(0, 1) < self.p[
                                'random_change_suppliers'] else 0
                            if self.personal_experience[good][supplier][1] < roll:
                                self.suppliers[good].remove(supplier)

                self.find_suppliers(good)
            self.make_purchases(good)

        if (not self.good) and self.criminal_needs:
            good = self.criminal_needs.pop(False)
            for i in range(self.criminal_multiplier):
                self.check_and_replace_former_criminal_supplier(good)
                self.make_criminal_purchases(good)

    def set_supplier(self, supplier,good, product=None):
        self.suppliers[good].append(supplier)
        if product is not None:
            if not supplier in self.product_details:
                self.product_details[supplier]= {}
            self.product_details[supplier][good]= product


    def set_criminal_supplier(self, supplier,good, product=None):
        self.criminal_suppliers[good].append(supplier)
        if product is not None:
            if not supplier in self.criminal_product_details:
                self.criminal_product_details[supplier]= {}
            self.criminal_product_details[supplier][good]= product

    # def remove_supplier(self, supplier,good, product=None):
    #     self.suppliers[good].remove(supplier)
    #     if product is not None and supplier in self.product_details:
    #         self.product_details[supplier].pop(good,None)


    def find_suppliers(self,good):

        # bad agents get their suppliers in the step at the start of the day, but because good
        # agents must continually judge their suppliers, they choose them after every transaction.
        # if the agent is good,
        #     go get a blacklist from experiences.
        #     if there is norep system,
        #         if the agent is not open to new experiences
        #             the agent first attempts to find someone it knows filtering out the blacklist
        #         if no supplier yet
        #             the agent chooses randomly from the suppliers excluding the blacklist,
        #             (without regard to whether its known)
        #     else if there is a rep system,
        #         the agent chooses according to choice algorithm but excluding the blacklist.
        #         which is a set that goes into the routine choose with threshold.
        #         if there is no thing in there, the agent chooses as if there were no rep system
        #         in the future we will include openness to new experience , with unopenness implying that the
        #         agent will return to old experiences.

        if len(self.suppliers[good]) < 1:
            # if the agent is good,
            #     go get a blacklist from experiences.

            #first go with the good ones you already know, in accord with your
            #openness to new experiences, and take it if its over the threshold

            threshold = max(self.fire_supplier_threshold,self.open_to_new_experiences)

            knowns_comfortable_with = {supplier:ratings_tuple[1] for supplier,ratings_tuple in self.personal_experience[good].items()
                      if (
                              supplier != self.unique_id
                              and supplier in self.model.suppliers[good]
                              # and good in self.model.agents[self.model.orig[int(supplier)]].supplying
                              and (not ratings_tuple[1] < threshold)
                              and (not self.supplier_inactive(supplier))
                      )
                      } if good in self.personal_experience and threshold < 0.95 else {}
            if len(knowns_comfortable_with)> 0:
            #   you have all the guys that you would stick to because they are so good. now choose
            # amongst those in proportion to their perceived goodness whether this is observed or not

                ratings_sum = sum([rating for key, rating in knowns_comfortable_with.items()])
                if ratings_sum > 0:
                    winner = None
                    roll = random.uniform(0, ratings_sum)
                    cumul = 0
                    for key, rating in knowns_comfortable_with.items():
                        if winner is None:
                            cumul = cumul + rating
                            if cumul > roll:
                                winner = key
                                product = self.product_details[key][good]
                                supplier_agent = self.model.agents[self.model.m[self.model.orig[key]]]
                                if product in supplier_agent.products[good]:
                                    product_quality = supplier_agent.products[good][product]['quality']
                                    if product_quality < threshold:
                                        product = self.choose_random_product(key, good)
                                        self.set_supplier(key,good,product)
                                    else:
                                        self.set_supplier(key,good)
                                    # if not open to a new experience just use last product they used for that good
                                    #self.suppliers[good].append(key)
                                else:
                                    product = self.choose_random_product(key, good)
                                    self.set_supplier(key,good,product)
                                    #print ("Product {0} is not in suppliers product list but is in product details".format(product))
            elif len(self.suppliers[good]) < 1:

                under_threshold = set([key for key, ratings_tuple in
                                       self.personal_experience[good].items(
                                       ) if ratings_tuple[1] <= self.fire_supplier_threshold]
                                      ) if good in self.personal_experience else set()
                if not self.p['observer_mode']:
                    new_supplier = self.choose_with_threshold(good, under_threshold)
                    if (new_supplier is not None):
                        parse = self.parse(new_supplier)
                        self.set_supplier(parse['agent'],parse['category'], parse['product'])

                        #self.suppliers[good].append(new_supplier)


                else :
            #     if there is norep system,
            #         if the agent is not open to new experiences
            #             the agent first attempts to find someone it knows filtering out the blacklist
                    if self.open_to_new_experiences < 0.95:#the roll is expensive
                        roll = random.uniform(0, 1)
                        if roll > self.open_to_new_experiences:
                            knowns = [supplier for supplier in self.personal_experience[good]
                                      if (
                                        supplier != self.unique_id
                                        and supplier in self.model.suppliers[good]
                                       # and good in self.model.agents[self.model.orig[int(supplier)]].supplying
                                        and (not supplier in under_threshold)
                                        and (not self.supplier_inactive(supplier))
                                      )
                                      ] if good in self.personal_experience else []
                            if len(knowns)> 0:
                                supplier_index = random.randint(0, len(knowns) - 1)#pick randomly over threshold
                                #self.suppliers[good].append(knowns[supplier_index])
                                self.set_supplier(knowns[supplier_index],good)


            #       if no supplier yet
            #             the agent chooses randomly from the suppliers excluding the blacklist,
            #             (without regard to whether its known)

                if len(self.suppliers[good]) < 1:
                    # youve got no choice but to try someone new at random
                    unknowns = [supplier for supplier in self.model.suppliers[good]
                                if (
                                        supplier != self.unique_id
                                        and (not supplier in under_threshold)
                                        and (not self.supplier_inactive(supplier))
                                )
                                ]

                    if len(unknowns)> 0:
                        supplier_index = random.randint(0, len(unknowns) - 1)
                        supplier = unknowns[supplier_index]
                        product = self.choose_random_product(supplier, good)
                        self.set_supplier(supplier, good, product)
                        #self.suppliers[good].append(unknowns[supplier_index])

    def choose_random_product(self, supplier, good):
        product = None
        if self.p['product_mode']:
            supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
            product = random.choice(list(supplier_agent.products[good]))
        return product

    def make_purchases(self, good):
        if len(self.suppliers[good]) > 0:
            merchandise_recieved = False
            supplier_idx = 0 if len(self.suppliers[good]) == 1 else random.randint(0, len(self.suppliers[good]) - 1)
            supplier = self.suppliers[good][supplier_idx]
            supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]

            consumer_id = self.unique_id  if supplier_agent.good else (
                self.incr_root_id(self.get_criminal_suppliers_increment (supplier) )
                if self.p["bad_consumers_change_names"] else self.unique_id)
            supplier_id =  supplier if supplier_agent.good else (
                supplier_agent.incr_root_id (self.get_criminal_suppliers_increment (supplier) ))

            #if not supplier_id in self.model.suppliers[good]:
                #self.add_new_supplier_with_good(supplier_id, good)

            amount = self.model.amount_distributions[good].rvs()
            if supplier in self.product_details and good in self.product_details[supplier]:
                product = self.product_details[supplier][good] if self.p['product_mode'] else None
                if product not in supplier_agent.products[good]:
                    product = self.choose_random_product(supplier, good)
                unit_price = (supplier_agent.products[good][product]['price'] if self.p['product_mode']
                              else self.model.agents[self.model.m[self.model.orig[supplier]]].supplying[good])
            else:
                product = -1
                unit_price = self.model.agents[self.model.m[self.model.orig[supplier]]].supplying[good]
            price = amount * unit_price

            good_string = "{0}.{1}".format(good, product) if self.p['product_mode'] else good
            reputation_to_entry = "{0}.{1}".format(supplier_id, good_string) if self.p['product_mode'] else supplier_id

            # two cases:
            # 1. payment without ratings:  child field populated with transid and parent left blank ,
            #	value has payment value, unit is payment unit.
            # 2. payment with ratings:  parent has id of payment while child has id of ranking,
            #	has ranking, unit blank, parent_value payment, parent_unit AGI

            # if supplier is not None:
            # if self.p['transactions_per_day'][0]== 1000:
            #    print ("agnet {0} repeat{1} purcahse of good{2}".format(self.unique_id, i, good))

            self.model.save_info_for_market_volume_report(self, self.model.orig[supplier], price)
            self.model.add_legit_transaction(supplier, price)
            perception, rating = self.rate_product(supplier, good, product) if self.p['product_mode'] else self.rate(
                supplier)
            if self.p['scam_goods_satisfy_needs'] or self.model.agents[self.model.m[self.model.orig[supplier]]].good:
                merchandise_recieved = True
            self.update_personal_experience(good, supplier_id, perception)
            self.model.add_good_consumer_rating(perception)

            quality = supplier_agent.products[good][product]["quality"]
            black_market = supplier_agent.products[good][product]["black_market"]
            self.model.add_organic_buy(price,quality,black_market)
            if self.p['product_mode']:
                self.record_shopping(supplier_agent,supplier, good, product,rating)
                supplier_agent.note_product_purchase(good, product, amount)

            # if self.p['include_ratings']:
            if self.p['include_ratings'] and rating and rating != "":
                self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                         price, good_string, self.p['types_and_units']['rating'],
                                                         rating=rating, type='rating')
                if not product in self.historic_products[good]:
                    self.model.send_trade_to_reputation_system(consumer_id, reputation_to_entry,
                                                               price, good, self.p['types_and_units']['rating'],
                                                               rating=rating, type='rating')

                    self.historic_products[good].add(product)

            elif not self.p['include_ratings']:
                self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                         price, good_string, self.p['types_and_units']['payment'])
                self.model.send_trade_to_reputation_system(consumer_id, reputation_to_entry,
                                                           price, good, self.p['types_and_units']['payment'])

            if (self.p['suppliers_are_consumers'] or len(self.supplying) < 1):

                for good1, supplierlist in self.suppliers.items():

                    bad_suppliers = [supplier for supplier in supplierlist if (
                            good1 in self.personal_experience and supplier in self.personal_experience[good1]
                            and len(self.personal_experience[good1][supplier]) > 0 and
                            self.personal_experience[good1][supplier][1] < self.fire_supplier_threshold)]
                    for bad_supplier in bad_suppliers:
                        supplierlist.remove(bad_supplier)

            if merchandise_recieved:  # comment out for unrestricted
                self.days_until_shop[good] = self.shopping_pattern[good]
                self.cobb_douglas_utilities[good] = self.cobb_douglas_utilities_original[good]

    def calculate_present_increment(self,supplier):

        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        our_day = self.model.daynum + supplier_agent.orig_scam_cycle_day
        generation_increment = (our_day // supplier_agent.scam_period) * self.p['num_users']
        return generation_increment

    def get_criminal_suppliers_increment(self,supplier):

        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        return supplier_agent.name_increment

    def exogenous_reputation_effect(self):
        if self.never_bought_scam:
            self.never_bought_scam = False
            roll = random.uniform (0,1)
            chance = self.p["chance_lose_external_reputation_each_scam"]
            if roll < chance:
                self.exogenous_reputation = False
                self.model.add_decision_to_scam()

    def make_criminal_purchases(self,good):
        if (not self.good) and good in self.criminal_suppliers and len(self.criminal_suppliers[good]) > 0:
            supplier_idx = 0 if len(self.criminal_suppliers[good]) == 1 else random.randint(0, len(self.criminal_suppliers[good]) - 1)
            supplier = self.criminal_suppliers[good][supplier_idx]

            supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]

            consumer_id = (self.incr_root_id(self.get_criminal_suppliers_increment (supplier) )
                           if self.p["bad_consumers_change_names"] else self.unique_id)
            supplier_id =  (self.model.orig[supplier] if supplier_agent.good else
                            supplier_agent.incr_root_id(self.get_criminal_suppliers_increment(supplier)))
                            #self.get_criminal_suppliers_increment (supplier) + self.model.orig[supplier])
            if self.p["bad_consumers_change_names"]:
                self.model.orig[consumer_id] = self.unique_id
            self.model.orig[supplier_id] = self.model.orig[supplier]
            #if not supplier_id in self.model.suppliers[good]:
                #self.add_new_supplier_with_good(supplier_id, good)

            amount = self.model.criminal_amount_distributions[good].rvs()
            if supplier in self.criminal_product_details and good in self.criminal_product_details[supplier]:
                product = self.criminal_product_details[supplier][good]if self.p['product_mode'] else None
                if product not in supplier_agent.products[good]:
                    product = self.choose_random_product(supplier, good)
                unit_price = (supplier_agent.products[good][product]['price']if self.p['product_mode']
                              else self.model.agents[self.model.m[self.model.orig[supplier]]].supplying[good])
            else:
                product = -1
                unit_price = self.model.agents[self.model.m[self.model.orig[supplier]]].supplying[good]
            price = amount * unit_price


            good_string = "{0}.{1}".format(good,product) if self.p['product_mode'] else good
            reputation_to_entry = "{0}.{1}".format(supplier_id,good_string) if self.p['product_mode'] else supplier_id


            if not self.supplier_inactive(supplier_id):
                self.model.save_info_for_market_volume_report(self, self.model.orig[supplier], price)
                supplier_agent.exogenous_reputation_effect()
                self.model.add_bought_review()
                supplier_agent.products[good][product]["black_market"] = True

                if self.model.error_log:
                    self.model.error_log.write("supplier {0} purchased amount {1} category {2} product {3} scam review from consumer {4}\n".format(
                    supplier_id,amount,good,product,self.unique_id))

               # print( "supplier {0} purchased amount {1} category {2} product {3} scam review from consumer {4}".format(
                 #       supplier_id, amount, good, product, self.unique_id))
                rating = ""
                if self.p['include_ratings']:
                    rating = self.best_rating() if (
                            random.uniform(0, 1) < self.p['criminal_chance_of_rating']) else self.p[
                        'non_rating_val']

                    self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                             price, good_string, self.p['types_and_units']['rating'],
                                                             rating=rating, type='rating', scam = True)
                    self.model.send_trade_to_reputation_system(consumer_id, reputation_to_entry,
                                                               price, good, self.p['types_and_units']['rating'],
                                                               rating=rating, type='rating')

                    self.historic_products[good].add(product)
                else:
                    self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                             price, good_string, self.p['types_and_units']['payment'], scam = True)
                    self.model.send_trade_to_reputation_system(consumer_id, reputation_to_entry,
                                                               price, good, self.p['types_and_units']['payment'])

                if self.p['product_mode']:
                    self.record_shopping(supplier_agent, supplier,good, product,rating)
                    supplier_agent.note_product_purchase(good,product,amount,scam=True)
                    commission = supplier_agent.products[good][product]["production_cost"] + \
                                 supplier_agent.products[good][product]["price"]
                    quality = supplier_agent.products[good][product]["quality"]
                    self.model.add_sponsored_buy(price, quality, commission)
            self.criminal_days_until_shop[good] = self.criminal_shopping_pattern[good]
            self.criminal_cobb_douglas_utilities[good] = self.cobb_douglas_utilities_original[good]


    def best_rating(self):
        dd = self.p['ratings_goodness_thresholds']
        best = next(reversed(dd))
        return best

    def get_initial_exogenous_reputation(self):
        evidence = {}
        result = self.model.roll_bayes(evidence, "supplier_exogenousReputation")
        exogenousReputation = True if result == "supplier_exogenousReputation" else False
        return exogenousReputation

    def leaves_rating(self,rating):
        evidence = {}
        evidence["product_quality"] = self.p['num_stars'][rating]
        evidence['PLRo_overall'] = self.p["PLRo_overall"]
        result = self.model.roll_bayes(evidence, "PLRo")
        leaves = True if result == "PLRo" else False
        return leaves

    def rate_product(self, supplier, category = None, product = None):
        # Rating is based on how bad the agent is in actuality, and then run through a perception which
        # puts some randomness in the ratings.  This is only for good rateds

        bias = self.model.rating_perception_distribution.rvs(
            ) if self.model.agents[self.model.m[self.model.orig[supplier]]].good else 0

        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        perception = (supplier_agent.products[category][product]['quality']
            if self.p['product_mode'] and product in supplier_agent.products[category] else supplier_agent.goodness)
        perception += bias
        #if not self.good:
            #perception = 1-perception

        rating = None
        dd = self.p['ratings_goodness_thresholds']
        for rating_val, threshold in dd.items():
            if (perception < threshold or threshold == dd[next(reversed(dd))])and rating is None:
                rating = rating_val

        if rating is not None:
            leaves_rating = self.leaves_rating(rating)
            if (not leaves_rating):
                rating = self.p['non_rating_val']

        if perception > 1:
            perception = 1
        if perception < 0:
            perception = 0

        return (perception,rating)

    def rate(self, supplier, category = None, product = None):
        # Rating is based on how bad the agent is in actuality, and then run through a perception which
        # puts some randomness in the ratings.  This is only for good rateds

        bias = self.model.rating_perception_distribution.rvs(
            ) if self.model.agents[self.model.m[self.model.orig[supplier]]].good else 0

        supplier_agent = self.model.agents[self.model.m[self.model.orig[supplier]]]
        perception = (supplier_agent.products[category][product]['quality']
            if self.p['product_mode'] and product in supplier_agent.products[category] else supplier_agent.goodness)
        perception += bias
        #if not self.good:
            #perception = 1-perception

        roll = random.uniform (0,1)
        rating = None

        dd = self.p['ratings_goodness_thresholds']
        for rating_val, threshold in dd.items():
            if (perception < threshold or threshold == dd[next(reversed(dd))])and rating is None:
                rating = rating_val
        if (rating is None or
                (self.model.agents[self.model.m[self.model.orig[supplier]]].good and roll > self.p['chance_of_rating_good2good']) or
                ((not self.model.agents[self.model.m[self.model.orig[supplier]]].good) and roll > self.p['chance_of_rating_good2bad'])
            ):
            rating = self.p['non_rating_val']

        return (perception,rating)


