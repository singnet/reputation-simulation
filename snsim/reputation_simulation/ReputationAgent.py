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

        self.initialize_products() if self.p['product_mode'] else {}


    def switch_project(self,category,product):
        #first check and see if it would make a difference in the personal name and dont change if it wouldnt,
        # but  once it makes a difference, test and decide
        evidence = {}
        evidence['product_profit'] = 'product_profit' if self.product_profit_positive(category,product) else 'product_profitNot'
        evidence['product_min_score'] = 'product_min_score' if self.product_min_score(category,product) else 'product_min_scoreNot'
        result = self.model.roll_bayes(evidence,"product_switch")
        change = True if result == "product_switch" else False
        return change

    def check_and_remove_products (self):
        for category, productdict in self.products.items():
            for product, _ in productdict.items():
                if self.switch_project(category,product ):
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
            self.products[category][pid]["black_market"] = not self.good
            self.products[category][pid]["last_sold"]= 0
            self.products[category][pid]["last_income"]= 0
            self.products[category][pid]["last_cost"]= 0
            self.products[category][pid]["total_sold"]= 0
            self.products[category][pid]["total_income"]= 0
            self.products[category][pid]["total_cost"]= 0
            self.products[category][pid]["initialization"]= True
            self.initialize_criminal_product_ring(category,pid)

    def initialize_products(self):
        self.products = {}
        # create products for every category I am selling
        # attrs:  price, production_cost, quality, black_market, last_sold, last_cost, avg_sold, avg_cost
        for category, _ in self.supplying.items():
            num_products = int(round(self.model.num_products_supplied_distribution.rvs()))
            self.products[category] = {}
            for _ in range(num_products):
                pid = self.model.next_product_id()
                self.products[category][pid] = {}
                self.products[category][pid] ["product_id"]= pid
                self.products[category][pid]["price"]= int(round(self.model.price_distributions[category].rvs(
                    ) if self.good else self.model.criminal_price_distributions[category].rvs() ))
                self.products[category][pid]["quality"]=(self.goodness +
                                                        self.model.quality_deviation_from_supplier_distribution.rvs())
                self.products[category][pid]["production_cost"] = (self.products[category][pid]["price"] *
                                                                       self.products[category][pid]["quality"])/2
                self.products[category][pid]["black_market"] = not self.good
                self.products[category][pid]["last_sold"]= 0
                self.products[category][pid]["last_income"]= 0
                self.products[category][pid]["last_cost"]= 0
                self.products[category][pid]["total_sold"]= 0
                self.products[category][pid]["total_income"]= 0
                self.products[category][pid]["total_cost"]= 0
                self.products[category][pid]["initialization"]= True
                self.initialize_criminal_product_ring(category,pid)


    def reset_daily_purchase_stats(self):
        for category,cdict in self.products.items():
            for pid,pdict in cdict.items():
                pdict["last_sold"]= 0
                pdict["last_income"]= 0
                pdict["last_cost"]= 0


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
        supplier_agent = self.model.agents[self.model.orig[supplier]]
        testset = self.historic_products[good] if good in self.historic_products else set()
        if supplier_agent.get_available_criminal_product(good,testset) is None:
            supplier_has_available_products = False
        return supplier_has_available_products



    def get_available_criminal_product(self,good,previously_bought):
        available = []
        if good in self.hiring_product:
            for product,hiring  in self.hiring_product[good].items():
                if (hiring and
                        len(self.criminal_consumers_product[good][product])
                        < min(self.num_criminal_consumers_product[good][product],self.p['max_scam_purchases']) and
                        not product in previously_bought):
                    available.append(product)
        random_available = random.choice(available) if len(available)else None
        return random_available

    def needs_criminal_consumer(self, supplier,good):
        needs = False
        supplier_agent = self.model.agents[self.model.orig[supplier]]
        if self.p['product_mode']:
            if good in supplier_agent.hiring_product:
                for product,hiring  in supplier_agent.hiring_product[good].items():
                    if hiring and len(supplier_agent.criminal_consumers_product[good][product]
                                      ) < min(supplier_agent.num_criminal_consumers_product[good][product],self.p['max_scam_purchases']):
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
            taken = True if len(suppliers) else False
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
                supplier_agent = self.model.agents[self.model.orig[supplier]]
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
                supplier_agent = self.model.agents[self.model.orig[supplier]]
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
        if len(self.model.ranks):
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



    def avg_score_over_threshold(self):
        avg_rank = self.model.get_avg_rank(self.unique_id)
        over = True if avg_rank > self.reputation_system_threshold else False
        return over

    def change_generations(self):
        #first check and see if it would make a difference in the personal name and dont change if it wouldnt,
        # but  once it makes a difference, test and decide
        evidence = {}
        evidence['supplier_profit'] = 'supplier_profit' if self.avg_profit_positive() else 'supplier_profitNot'
        evidence['supplier_min_score'] = 'supplier_min_score' if self.avg_score_over_threshold() else 'supplier_min_scoreNot'
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
                    supplier_agent = self.model.agents[self.model.orig[supplier]]
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


    def step(self):
        #print ("first line of ReputationAgent step")
        #first increment the  number of days that have taken place to shop for all goods
        #tempDict = {good: days-1 for good, days in self.days_until_shop.items() if days > 0 }
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
            if self.model.reputation_system and self.p['average_reputation_system_threshold']:
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

        if self.p['choice_method'] == "thresholded_random":

            # thresholded random, using the reputation system AND past experience

            over_threshold = [agent for agent, rating in self.model.ranks.items()
                              if
                              rating > self.reputation_system_threshold
                              and int(agent) in self.model.suppliers[good]
                              #and good in self.model.agents[self.model.orig[int(agent)]].supplying
                              and int(agent) not in under_threshold
                              and not self.supplier_inactive(int(agent))]
            if len(over_threshold):
                roll = np.random.randint(0, len(over_threshold))
                winner = int(over_threshold[roll])
        elif self.p['choice_method'] == "winner_take_all":


            non_criminal_experiences = {int(agent): rating for agent, rating in self.model.ranks.items()
                                        if
                                        rating > self.reputation_system_threshold
                                        and int(agent) in self.model.suppliers[good]
                                        #and good in self.model.agents[self.model.orig[int(agent)]].supplying
                                        and not self.supplier_inactive(int(agent))}
            sorted_suppliers = sorted(non_criminal_experiences.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_suppliers):
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
        parse['agent']= int(agent_string_split[0])  #if self.p['product_mode'] else int(agent_string)

        parse['category']= agent_string_split[1]if len(agent_string_split)==3 else None
        parse['product']= int(agent_string_split[2])if len(agent_string_split)==3 else None
        return parse



    def supplier_inactive(self, supplier):
        #inactive if agent is in SAP or if agents orig is in sap, or if the agent is hiding its name.
        #now, inactive means just inactive to bad agents.  The supplier is always active to good agents
        if self.good:
            inactive = False
        else:
            inactive = True

            supplier_agent = self.model.agents[self.model.orig[supplier]]
            if supplier_agent.good:
                inactive = False
            else:
                generation_increment = self.get_criminal_suppliers_increment(supplier)
                #if (supplier_agent.scam_cycle_day >= supplier.scam_inactive_period):
                scam_active_period = supplier_agent.scam_period - supplier_agent.scam_inactive_period
                if (supplier_agent.scam_cycle_day < scam_active_period):
                    inactive = False
                if (supplier != self.model.orig[supplier]+generation_increment) :
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
        rank = self.model.ranks[rankstring] if rankstring in self.model.ranks else -1
        score = True if rank > self.reputation_system_threshold else False
        return score

    def initialize_criminal_product_ring(self, good, product):
        if self.p['criminal_suppliers_purchase_reviewers']:

            if not good in self.hiring_product:
                self.hiring_product[good]= {}
            self.hiring_product[good][product] = True

            if not good in self.criminal_consumers_product:
                self.criminal_consumers_product[good]= {}
            self.criminal_consumers_product[good][product] = set()

            if self.products[good][product]['initialization']:
                self.products[good][product]['initialization']=False
                macroview = self.model.config['parameters']['macro_view']
                if macroview and self.model.config['macro_views'][macroview]['supplement'][
                         "overwrite_criminal_agent_ring_size"] == "market_research":
                    purch = self.model.config['macro_views'][macroview]['supplement']['purch']
                    profit_margin = self.model.config['macro_views'][macroview]['supplement']['profit_margin']
                    beta = self.model.config['macro_views'][macroview]['supplement']['beta']
                    alpha = self.model.config['macro_views'][macroview]['supplement']['alpha']

                    #for good,price in self.supplying.items():
                    #for product in self.product_details[good]:
                    price = self.products[good][product]['price']
                    if not good in self.num_criminal_consumers_product:
                        self.num_criminal_consumers_product[good]= {}
                    self.num_criminal_consumers_product[good][product] = 0 if price <= 0 else (
                        (purch/math.pow(price,beta))*
                        ((price*profit_margin)/(price + pow(price,alpha)))
                        )
                else:
                    self.num_criminal_consumers_product[good][product] = int(round(self.model.criminal_agent_ring_size_distribution.rvs()))
            else:
                evidence = {}
                evidence['product_profit'] = 'product_profit' if self.product_profit_positive(good, product) else 'product_profitNot'
                evidence['product_best_score'] = 'product_best_score' if self.product_best_score(good, product) else 'product_best_scoreNot'
                winner = self.model.roll_bayes(evidence, "num_bought_scams")

                change = self.p["change_num_scams_incr"] if winner == 'product_scam_higher' else (
                    -self.p["change_num_scams_incr"] if winner == 'product_scam_lower' else (0))
                self.num_criminal_consumers_product[good][product] += change
                if self.num_criminal_consumers_product[good][product] < self.p['min_scam_purchases']:
                    self.num_criminal_consumers_product[good][product] = self.p['min_scam_purchases']
                elif self.num_criminal_consumers_product[good][product] > self.p['max_scam_purchases']:
                    self.num_criminal_consumers_product[good][product] = self.p['max_scam_purchases']
                #if self.num_criminal_consumers_product[good][product] > 0 :
                if self.model.error_log:
                    self.model.error_log.write("supplier {4} product {0} category {1} num to buy reviews changes {2} to {3}\n".format(
                        product,good,change,self.num_criminal_consumers_product[good][product],self.unique_id))

                #print("supplier {4} product {0} category {1} num to buy reviews changes {2} to {3}".format(
                 #   product,good,change,self.num_criminal_consumers_product[good][product],self.unique_id))
        else:
            self.num_criminal_consumers_product[good][product] = 0

    def initialize_criminal_ring(self, good):
        self.hiring[good]= True
        macroview = self.model.config['parameters']['macro_view']
        if macroview and self.model.config['macro_views'][macroview]['supplement'][
                 "overwrite_criminal_agent_ring_size"] == "market_research":
            purch = self.model.config['macro_views'][macroview]['supplement']['purch']
            profit_margin = self.model.config['macro_views'][macroview]['supplement']['profit_margin']
            beta = self.model.config['macro_views'][macroview]['supplement']['beta']
            alpha = self.model.config['macro_views'][macroview]['supplement']['alpha']
            #error why goods done multiple times??
            #for good,price in self.supplying.items():
            price = self.supplying[good]

            self.num_criminal_consumers[good] = 0 if price <= 0 else (
                (purch/math.pow(price,beta))*
                ((price*profit_margin)/(price + pow(price,alpha)))
                )

    def clear_products(self):

        for good,_ in self.supplying.items():
            self.criminal_consumers_product[good]={}
            self.num_criminal_consumers_product[good]={}
            self.hiring_product[good]= {}
            self.products[good] = {}

    def initialize_criminal_product_rings(self):
        for good, pdict in self.products.items():
            for product, _ in pdict.items():
                self.initialize_criminal_product_ring(good, product)


    def change_name(self):
        #move stuff here to find out what name to change to

        calc = self.calculate_present_increment(self.unique_id)
        self.name_increment = calc
        supplier = self.get_criminal_suppliers_increment(self.unique_id) + self.unique_id
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
                scam_periods_so_far = (self.model.daynum // self.scam_period) + 1
                for i in range(scam_periods_so_far):
                    alias = (i * self.p['num_users']) + self.model.orig[supplier]
                    if alias != supplier and alias in self.model.suppliers[good]:
                        self.model.suppliers[good].remove(alias)
                        if not self.good:
                            self.model.criminal_suppliers[good].remove(alias)
                        self.model.finalize_rank(alias)
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
                      } if good in self.personal_experience else {}
            if len(knowns_comfortable_with):
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
                                supplier_agent = self.model.agents[self.model.orig[key]]
                                product_quality = supplier_agent.products[good][product]['quality']
                                if product_quality < threshold:
                                    product = self.choose_random_product(key, good)
                                    self.set_supplier(key,good,product)
                                else:
                                    self.set_supplier(key,good)
                                # if not open to a new experience just use last product they used for that good
                                #self.suppliers[good].append(key)

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
                            if len(knowns):
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

                    if len(unknowns):
                        supplier_index = random.randint(0, len(unknowns) - 1)
                        supplier = unknowns[supplier_index]
                        product = self.choose_random_product(supplier, good)
                        self.set_supplier(supplier, good, product)
                        #self.suppliers[good].append(unknowns[supplier_index])

    def choose_random_product(self, supplier, good):
        product = None
        if self.p['product_mode']:
            supplier_agent = self.model.agents[self.model.orig[supplier]]
            product = random.choice(list(supplier_agent.products[good]))
        return product

    def make_purchases(self, good):
        if len(self.suppliers[good]) > 0:
            merchandise_recieved = False
            supplier_idx = 0 if len(self.suppliers[good]) == 1 else random.randint(0, len(self.suppliers[good]) - 1)
            supplier = self.suppliers[good][supplier_idx]
            supplier_agent = self.model.agents[self.model.orig[supplier]]

            consumer_id = self.unique_id  if supplier_agent.good else (
                self.get_criminal_suppliers_increment (supplier)  + self.unique_id if self.p["bad_consumers_change_names"] else self.unique_id)
            supplier_id =  supplier if supplier_agent.good else (
                self.get_criminal_suppliers_increment (supplier) + self.model.orig[supplier])

            #if not supplier_id in self.model.suppliers[good]:
                #self.add_new_supplier_with_good(supplier_id, good)

            amount = self.model.amount_distributions[good].rvs()
            if supplier in self.product_details and good in self.product_details[supplier]:
                product = self.product_details[supplier][good] if self.p['product_mode'] else None
                if product not in supplier_agent.products[good]:
                    product = self.choose_random_product(supplier, good)
                unit_price = (supplier_agent.products[good][product]['price'] if self.p['product_mode']
                              else self.model.agents[self.model.orig[supplier]].supplying[good])
            else:
                product = -1
                unit_price = self.model.agents[self.model.orig[supplier]].supplying[good]
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
            perception, rating = self.rate_product(supplier, good, product) if self.p['product_mode'] else self.rate(
                supplier)
            if self.p['scam_goods_satisfy_needs'] or self.model.agents[self.model.orig[supplier]].good:
                merchandise_recieved = True
            self.update_personal_experience(good, supplier_id, perception)
            self.model.add_organic_buy(price,perception)
            if self.p['product_mode']:
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
        supplier_agent = self.model.agents[self.model.orig[supplier]]
        our_day = self.model.daynum + supplier_agent.orig_scam_cycle_day
        generation_increment = (our_day // supplier_agent.scam_period) * self.p['num_users']
        return generation_increment

    def get_criminal_suppliers_increment(self,supplier):

        supplier_agent = self.model.agents[self.model.orig[supplier]]
        return supplier_agent.name_increment


    def make_criminal_purchases(self,good):
        if (not self.good) and good in self.criminal_suppliers and len(self.criminal_suppliers[good]) > 0:
            supplier_idx = 0 if len(self.criminal_suppliers[good]) == 1 else random.randint(0, len(self.criminal_suppliers[good]) - 1)
            supplier = self.criminal_suppliers[good][supplier_idx]

            supplier_agent = self.model.agents[self.model.orig[supplier]]

            consumer_id =  self.get_criminal_suppliers_increment (supplier)  + self.unique_id if self.p["bad_consumers_change_names"] else self.unique_id
            supplier_id =  self.model.orig[supplier] if supplier_agent.good else self.get_criminal_suppliers_increment (supplier) + self.model.orig[supplier]
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
                              else self.model.agents[self.model.orig[supplier]].supplying[good])
            else:
                product = -1
                unit_price = self.model.agents[self.model.orig[supplier]].supplying[good]
            price = amount * unit_price


            good_string = "{0}.{1}".format(good,product) if self.p['product_mode'] else good
            reputation_to_entry = "{0}.{1}".format(supplier_id,good_string) if self.p['product_mode'] else supplier_id


            if not self.supplier_inactive(supplier_id):
                self.model.save_info_for_market_volume_report(self, self.model.orig[supplier], price)
                if self.model.error_log:
                    self.model.error_log.write("supplier {0} purchased amount {1} category {2} product {3} scam review from consumer {4}\n".format(
                    supplier_id,amount,good,product,self.unique_id))

               # print( "supplier {0} purchased amount {1} category {2} product {3} scam review from consumer {4}".format(
                 #       supplier_id, amount, good, product, self.unique_id))
                if self.p['product_mode']:
                    supplier_agent.note_product_purchase(good,product,amount,scam=True)
                    commission = supplier_agent.products[good][product]["production_cost"] + \
                                 supplier_agent.products[good][product]["price"]
                    quality = supplier_agent.products[good][product]["quality"]
                    self.model.add_sponsored_buy(price, quality, commission)
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

            self.criminal_days_until_shop[good] = self.criminal_shopping_pattern[good]
            self.criminal_cobb_douglas_utilities[good] = self.cobb_douglas_utilities_original[good]


    def best_rating(self):
        dd = self.p['ratings_goodness_thresholds']
        best = next(reversed(dd))
        return best


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
            ) if self.model.agents[self.model.orig[supplier]].good else 0

        supplier_agent = self.model.agents[self.model.orig[supplier]]
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

        return (perception,rating)

    def rate(self, supplier, category = None, product = None):
        # Rating is based on how bad the agent is in actuality, and then run through a perception which
        # puts some randomness in the ratings.  This is only for good rateds

        bias = self.model.rating_perception_distribution.rvs(
            ) if self.model.agents[self.model.orig[supplier]].good else 0

        supplier_agent = self.model.agents[self.model.orig[supplier]]
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
                (self.model.agents[self.model.orig[supplier]].good and roll > self.p['chance_of_rating_good2good']) or
                ((not self.model.agents[self.model.orig[supplier]].good) and roll > self.p['chance_of_rating_good2bad'])
            ):
            rating = self.p['non_rating_val']

        return (perception,rating)


