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
        self.reputation_system_threshold = self.model.reputation_system_threshold_distribution.rvs()
        self.forget_discount = self.model.forget_discount_distribution.rvs()
        self.open_to_new_experiences = self.model.open_to_new_experiences_distribution.rvs()
        self.personal_experience = OrderedDict()
        tuplist = [(good, 0) for good, chance in self.p["chance_of_supplying"].items()]
        self.days_until_shop =   OrderedDict(tuplist)
        self.shopping_pattern = OrderedDict()
        self.num_criminal_consumers = OrderedDict()
        self.criminal_consumers = OrderedDict()

        cumulative = 0
        self.cobb_douglas_utilities = OrderedDict()
        self.needs = []
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



        tuplist = [(good,[]) for good, chance in self.p["chance_of_supplying"].items()]
        self.suppliers = OrderedDict(tuplist)


        if self.good:
            for good, needrv in self.model.need_cycle_distributions.items():
                self.shopping_pattern[good] = needrv.rvs()
                if self.p["randomize_initial_needs"]:
                    self.days_until_shop[good] = random.randint(0,round(self.shopping_pattern[good]) )
        else:
            if supply_list is not None and len(supply_list) > 0:
                #There will be overlap in the criminal rings that criminals go to
                self.num_criminal_consumers = {good:int(round(self.model.criminal_agent_ring_size_distribution.rvs())) for good in supply_list}
                self.criminal_consumers = {good:set() for good in supply_list}
                self.scam_cycle_day = random.randint(0,self.p['scam_parameters']["scam_period"] -1)
                self.orig_scam_cycle_day = self.scam_cycle_day
            for good, needrv in self.model.criminal_need_cycle_distributions.items():
                self.shopping_pattern[good] = needrv.rvs()

    def needs_criminal_consumer(self, supplier,good):
        needs = False
        supplier_agent = self.model.agents[self.model.orig[supplier]]
        if len(supplier_agent.criminal_consumers[good]) < supplier_agent.num_criminal_consumers[good]:
            needs = True
        return needs

    def taken(self):
        taken = False
        if not self.good:
            suppliers = [supplier  for good, supplierlist in self.suppliers.items()for supplier in supplierlist]
            taken = True if len(suppliers) else False
        return taken

    def adopt_criminal_supplier(self, good):
        supplier = None
        possible_suppliers = [supplier for supplier in self.model.criminal_suppliers[good] if (
                supplier != self.unique_id and
                self.needs_criminal_consumer(supplier, good))  and
                not self.supplier_inactive(supplier) ]
        if len(possible_suppliers) > 0:
            supplier = possible_suppliers[random.randint(0,len(possible_suppliers)-1)]
            supplier_agent = self.model.agents[self.model.orig[supplier]]
            supplier_agent.criminal_consumers[good].add(self.unique_id)
            self.suppliers[good].append(supplier)

    def clear_supplierlist (self, good,supplierlist):
        if not self.good:
            for supplier in supplierlist:
                supplier_agent = self.model.agents[self.model.orig[supplier]]
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


    def step(self):
        #print ("first line of ReputationAgent step")
        #first increment the  number of days that have taken place to shop for all goods
        #tempDict = {good: days-1 for good, days in self.days_until_shop.items() if days > 0 }
        tempDict = {good: days-1 for good, days in self.days_until_shop.items()  }
        self.days_until_shop.update(tempDict)


        if (self.p['suppliers_are_consumers'] or len(self.supplying)>= 1):
            self.scam_cycle_day += 1
            if (self.scam_cycle_day % self.p['scam_parameters']['scam_period'])==0:
                self.scam_cycle_day = 0
        # kick out suppliers. See how many trades you will make.
        # initialize todays goods
        # go through and make a list of needs in order

        if len(self.supplying)>= 1 and not self.good:
            our_day = self.model.daynum + self.orig_scam_cycle_day
            generation_increment = (our_day // self.p['scam_parameters']['scam_period']) * self.p['num_users']

            supplier_id = generation_increment + self.unique_id
            self.model.orig[supplier_id] = self.unique_id
            for good,price in self.supplying.items():
                if not supplier_id in self.model.suppliers[good]:
                    self.add_new_supplier(supplier_id, good)

        if (self.p['suppliers_are_consumers'] or len(self.supplying)< 1):

            if (not self.good) and (not self.taken()):
                # for good, supplierlist in self.suppliers.items():
                #     for supplier in supplierlist:
                #         supplier_agent = self.model.agents[self.model.orig[supplier]]
                #         if self.p['random_change_suppliers'] == 1.0 or self.supplier_inactive(supplier):
                #             if self.unique_id in supplier_agent.criminal_consumers[good]:
                #                 supplier_agent.criminal_consumers[good].remove(self.unique_id)
                #                 supplierlist.remove(supplier)
                #         else:
                #             roll = random.uniform(0,1)
                #             if roll < self.p['random_change_suppliers']:
                #                 if self.unique_id in supplier_agent.criminal_consumers[good]:
                #                     supplier_agent.criminal_consumers[good].remove(self.unique_id)
                #                     supplierlist.remove(supplier)

                for good, supplierlist in self.suppliers.items():
                    if len(supplierlist) < 1 and not self.taken():
                        self.adopt_criminal_supplier(good)

            #have agents start out with minute, non zero supplies of all goods, to make sure cobb douglas works
            tuplist = [(good, random.uniform(0.1,0.2)) for good, chance in self.p["chance_of_supplying"].items()]
            self.goods = OrderedDict(tuplist)
            num_trades =int(round(self.model.transactions_per_day_distribution.rvs(
                ) if self.good else self.model.criminal_transactions_per_day_distribution.rvs() ))
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

            if self.good:
                self.update_utilities()
                p = [v for v in list(self.cobb_douglas_utilities.values())if v > 0]
                n = len(p)
                keys = list(self.cobb_douglas_utilities.keys())[0:n]
                wants = list(np.random.choice(keys,n , p=p, replace=False))
                self.needs = [want for want in wants if self.days_until_shop[want] < 1]#todo: sorting is faster and almost the same
                if self.model.reputation_system and self.p['average_reputation_system_threshold']:
                    self.reputation_system_threshold = self.average_ranks()
            else:
                self.needs = [good for good, supplierlist in self.suppliers.items() if len(supplierlist) > 0]
            self.multiplier = 1
            if num_trades and  num_trades < len(self.needs):
                self.needs = self.needs[:num_trades]
            elif self.needs:
                self.multiplier = int(round(num_trades/len (self.needs)))


    def update_personal_experience(self, good, supplier, rating):
        #rating = float(rating_str)
        if not good in self.personal_experience:
            self.personal_experience[good]= OrderedDict()
        if supplier not in self.personal_experience[good] :
            self.personal_experience[good][supplier] = (1,rating)
        else:
            times_rated, past_rating = self.personal_experience[good][supplier]
            new_times_rated = times_rated + 1
            #now_factor = 1 + (1-self.forget_discount)
            #new_rating = ((times_rated * past_rating * self.forget_discount)  + (rating * now_factor))/new_times_rated
            new_rating = rating
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

                non_criminal_experiences = {int(agent): rating for agent, rating in self.model.ranks.items()
                                            if
                                            (rating > self.reputation_system_threshold)
                                            and int(agent) in self.model.suppliers[good]
                                            #and (good in self.model.agents[self.model.orig[int(agent)]].supplying )
                                            and (not int(agent) in under_threshold)
                                            and (not self.supplier_inactive(int(agent)))}
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

    def supplier_inactive(self, supplier):
        #inactive if agent is in SAP or if agents orig is in sap, or if the agent is hiding its name.
        inactive = True

        supplier_agent = self.model.agents[self.model.orig[supplier]]
        if supplier_agent.good:
            inactive = False
        else:
            our_day = self.model.daynum + supplier_agent.orig_scam_cycle_day
            generation_increment = (our_day // self.p['scam_parameters']['scam_period']) * self.p['num_users']
            if (supplier_agent.scam_cycle_day >= self.p['scam_parameters']['scam_inactive_period']):
                inactive = False
            if (supplier != self.model.orig[supplier]+generation_increment) :
                inactive = True

        return inactive

    def add_new_supplier(self, supplier, good):
        if not supplier in self.model.suppliers[good]:
            self.model.suppliers[good].add(supplier)
            self.model.initialize_rank(supplier)
            # if supplier != self.model.orig[supplier] and self.model.orig[supplier] in self.model.suppliers[good]:
            #     self.model.suppliers[good].remove(self.model.orig[supplier])
            scam_periods_so_far = (self.model.daynum // self.p['scam_parameters']['scam_period'])+1
            for i in range(scam_periods_so_far):
                alias = (i * self.p['num_users']) + self.model.orig[supplier]
                if alias != supplier and alias in self.model.suppliers[good]:
                    self.model.suppliers[good].remove(alias)
                    self.model.finalize_rank(alias)
            self.model.average_rank_history.flush()


    def choose_partners(self):
        #one good per each time you enter.  change this by looping on goods per day if you have more goods that
        #times that choose_partners is called.

        if self.needs:
            good = self.needs.pop(False)
            for i in range(self.multiplier):
                if self.good:

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

        if self.good and len(self.suppliers[good]) < 1:
            # if the agent is good,
            #     go get a blacklist from experiences.
            under_threshold = set([key for key, ratings_tuple in
                                   self.personal_experience[good].items(
                                   ) if ratings_tuple[1] <= self.fire_supplier_threshold]
                                  ) if good in self.personal_experience else set()
            if not self.p['observer_mode']:
                new_supplier = self.choose_with_threshold(good, under_threshold)
                if (new_supplier is not None):
                    self.suppliers[good].append(new_supplier)
            if self.p['observer_mode'] or len(self.suppliers[good]) < 1:
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
                            supplier_index = random.randint(0, len(knowns) - 1)
                            self.suppliers[good].append(knowns[supplier_index])


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
                        self.suppliers[good].append(unknowns[supplier_index])



    def make_purchases(self,good):
        if len(self.suppliers[good]) > 0:
            merchandise_recieved = False
            supplier_idx = 0 if len(self.suppliers[good]) == 1 else random.randint(0, len(self.suppliers[good]) - 1)
            supplier = self.suppliers[good][supplier_idx]

            supplier_agent = self.model.agents[self.model.orig[supplier]]
            our_day = self.model.daynum + supplier_agent.orig_scam_cycle_day
            generation_increment = (our_day // self.p['scam_parameters']['scam_period']) * self.p['num_users']

            consumer_id = self.unique_id if self.good else generation_increment + self.unique_id
            supplier_id = self.model.orig[supplier] if supplier_agent.good else generation_increment + self.model.orig[
                supplier]
            self.model.orig[consumer_id] = self.unique_id
            self.model.orig[supplier_id] = self.model.orig[supplier]
            if not supplier_id in self.model.suppliers[good]:
                self.add_new_supplier(supplier_id, good)

            amount = self.model.amount_distributions[good].rvs() if self.good else \
            self.model.criminal_amount_distributions[good].rvs()
            price = amount * self.model.agents[self.model.orig[supplier]].supplying[good]

            # two cases:
            # 1. payment without ratings:  child field populated with transid and parent left blank ,
            #	value has payment value, unit is payment unit.
            # 2. payment with ratings:  parent has id of payment while child has id of ranking,
            #	has ranking, unit blank, parent_value payment, parent_unit AGI

            # if supplier is not None:
            # if self.p['transactions_per_day'][0]== 1000:
            #    print ("agnet {0} repeat{1} purcahse of good{2}".format(self.unique_id, i, good))

            if self.good:
                self.model.save_info_for_market_volume_report(self, self.model.orig[supplier], price)
                perception, rating = self.rate(supplier)
                if self.model.agents[self.model.orig[supplier]].good:
                    merchandise_recieved = True
                self.update_personal_experience(good, supplier_id, perception)
                if self.p['include_ratings'] and rating and rating != "":
                    self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                             price, good, self.p['types_and_units']['rating'],
                                                             rating=rating, type='rating')

                    self.model.send_trade_to_reputation_system(consumer_id, supplier_id,
                                                               price, good, self.p['types_and_units']['rating'],
                                                               rating=rating, type='rating')
                elif not self.p['include_ratings']:
                    self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                             price, good, self.p['types_and_units']['payment'])
                    self.model.send_trade_to_reputation_system(consumer_id, supplier_id,
                                                               price, good, self.p['types_and_units']['payment'])
            else:

                if not self.supplier_inactive(supplier_id):
                    self.model.save_info_for_market_volume_report(self, self.model.orig[supplier], price)
                    if self.p['include_ratings']:
                        rating = self.best_rating() if (
                                random.uniform(0, 1) < self.p['criminal_chance_of_rating']) else self.p[
                            'non_rating_val']

                        self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                                 price, good, self.p['types_and_units']['rating'],
                                                                 rating=rating, type='rating')
                        self.model.send_trade_to_reputation_system(consumer_id, supplier_id,
                                                                   price, good, self.p['types_and_units']['rating'],
                                                                   rating=rating, type='rating')
                    else:
                        self.model.print_transaction_report_line(consumer_id, supplier_id,
                                                                 price, good, self.p['types_and_units']['payment'])
                        self.model.send_trade_to_reputation_system(consumer_id, supplier_id,
                                                                   price, good, self.p['types_and_units']['payment'])

            if (self.p['suppliers_are_consumers'] or len(self.supplying) < 1):

                if self.good:
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


    def best_rating(self):
        dd = self.p['ratings_goodness_thresholds']
        best = next(reversed(dd))
        return best

    def rate(self, supplier):
        # Rating is based on how bad the agent is in actuality, and then run through a perception which
        # puts some randomness in the ratings.  This is only for good rateds

        bias = self.model.rating_perception_distribution.rvs(
            ) if self.model.agents[self.model.orig[supplier]].good else 0
        perception = self.model.agents[self.model.orig[supplier]].goodness + bias
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


