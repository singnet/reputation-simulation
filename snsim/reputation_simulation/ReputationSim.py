import json
import os
import pickle
import sys
import re
import random
import numpy as np
np.set_printoptions(threshold=np.inf)
from collections import OrderedDict
import copy
import datetime as dt
import time
import operator
from scipy.stats import truncnorm
from pomegranate import *
import psyneulink as pnl

from ReputationAgent import ReputationAgent
from Adapters import Adapters
from ContinuousRankByGoodTests import ContinuousRankByGoodTests
from ContinuousRankTests import ContinuousRankTests
from DiscreteRankTests import DiscreteRankTests
from GoodnessTests import GoodnessTests
from MarketVolumeTests import MarketVolumeTests
from TransactionsTests import TransactionsTests
import math
#from reputation import Aigents
from random import shuffle
from reputation import AigentsAPIReputationService
from reputation.reputation_base_api import *
from reputation import PythonReputationService
#from reputation.reputation_service_api import PythonReputationService


from mesa import Model
from mesa.time import StagedActivation

class ReputationSim(Model):


    # def __new__(cls, *args, **kwargs):
    #     print ("first line of __new__")
    #     if kwargs['opened_config']:
    #         config = kwargs['study_path']
    #     else:
    #         with open(kwargs['study_path']) as json_file:
    #
    #             config = json.load(json_file, object_pairs_hook=OrderedDict)
    #
    #         # save the config with the output
    #
    #     print("calling super __init__")
    #     kwargs['seed']= config['parameters']['seed']
    #
    #
    #     return super(ReputationSim, cls).__new__(cls, *args, **kwargs)


    def __init__(self,study_path='study.json',rs=None,  opened_config= False):
       # print('First line of init in RepuationSim, study path is ${0}'.format(study_path))
        self.current_product_id = 0
        self.config = None
        self.study_path = study_path
        if opened_config:
            self.config = study_path
        else:
            with open(study_path) as json_file:

                self.config = json.load(json_file, object_pairs_hook=OrderedDict)

        self.reputation_system = self.make_reputation_system(self.config) if rs is None else rs
        if self.reputation_system is not None:
           self.initialize_reputation_system(self.config)

        #save the config with the output

        self.bayesian_network = self.get_bayesian_networks()

        self.id_pattern = re.compile(r'(\d+)\-(\d+)')
        self.transaction_numbers = []
        transaction_number = 0
        # print(json.dumps(config['ontology'], indent=2))
        self.parameters = self.config['parameters']
        #super().__init__(config['parameters']['seed'])
        super().__init__()
        super().reset_randomizer(self.parameters['seed'])

        #self.orig = {i:i for i in range(self.config['parameters']['num_users'])}
        self.orig = {}
        self.m = {}

        self.time = dt.datetime.now().isoformat()

        self.rank_history_heading = ""

        if not os.path.exists(self.parameters['output_path']):
        #   raise Exception('Directory {0} exists'.format(self.parameters['output_path']))
        #else:
            os.makedirs(self.parameters['output_path'])
        #filename = self.parameters['output_path'] + 'params_' + self.parameters['param_str'] + self.time[0:10] + '.json'
        filename = self.parameters['output_path'] + 'params_' + self.parameters['param_str'][:-1] + '.json'

        pretty = json.dumps(self.config, indent=2, separators=(',', ':'))
        with open(filename, 'w') as outfile:
            outfile.write(pretty)
        outfile.close()
        self.average_rank_history = self.average_rank_history()
        self.transaction_report = self.transaction_report()
        self.market_volume_report = self.market_volume_report()
        self.error_log = self.error_log() if self.parameters['error_log'] else None


        self.seconds_per_day = 86400


        tuplist = [(good,set()) for good, chance in self.parameters["chance_of_supplying"].items()]
        self.suppliers = OrderedDict(tuplist)

        tuplist = [(good, set()) for good, chance in self.parameters["criminal_chance_of_supplying"].items()]
        self.criminal_suppliers = OrderedDict(tuplist)

        self.individual_agent_market_volume ={}


        self.initial_epoch = self.get_epoch(self.parameters['initial_date'])
        self.final_epoch = self.get_epoch(self.parameters['final_date'])
        self.since = self.get_datetime(self.parameters['initial_date'])
        self.daynum = 0
        self.next_transaction = 0
        self.end_tick = self.get_end_tick()
        self.goodness_distribution = self.get_truncated_normal(*tuple(self.parameters['goodness']) )
        self.fire_supplier_threshold_distribution = self.get_truncated_normal(
            *tuple(self.parameters['fire_supplier_threshold']))
        self.reputation_system_threshold_distribution = self.get_truncated_normal(
            *tuple(self.parameters['reputation_system_threshold']))
        self.forget_discount_distribution = self.get_truncated_normal(
            *tuple(self.parameters['forget_discount']))
        self.criminal_transactions_per_day_distribution = self.get_truncated_normal(
            *tuple(self.parameters['criminal_transactions_per_day']))
        self.transactions_per_day_distribution = self.get_truncated_normal(
            *tuple(self.parameters['transactions_per_day']))
        self.sp_distribution = self.get_truncated_normal(
            *tuple(self.parameters['scam_parameters']['scam_period']))
        self.num_products_supplied_distribution = self.get_truncated_normal(
            *tuple(self.parameters['num_products_supplied']))
        self.quality_deviation_from_supplier_distribution = self.get_truncated_normal(
            *tuple(self.parameters['quality_deviation_from_supplier']))
        self.sip_distribution = self.get_truncated_normal(
            *tuple(self.parameters['scam_parameters']['scam_inactive_period']))
        self.criminal_agent_ring_size_distribution = self.get_truncated_normal(*tuple(self.parameters['criminal_agent_ring_size']) )
        self.open_to_new_experiences_distribution = self.get_truncated_normal(*tuple(self.parameters['open_to_new_experiences']) )
        self.criminal_goodness_distribution = self.get_truncated_normal(*tuple(self.parameters['criminal_goodness']) )

        self.rating_perception_distribution = self.get_truncated_normal(*tuple(self.parameters['rating_perception']) )
        self.cobb_douglas_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                            ) for good, statlist in self.parameters['cobb_douglas_utilities'].items()}
        self.price_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                            ) for good, statlist in self.parameters['prices'].items()}
        self.criminal_price_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                            ) for good, statlist in self.parameters['criminal_prices'].items()}
        self.need_cycle_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                            ) for good, statlist in self.parameters['need_cycle'].items()}
        self.criminal_need_cycle_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                            ) for good, statlist in self.parameters['criminal_need_cycle'].items()}
        self.amount_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                                                    ) for good, statlist in
                                    self.parameters['amounts'].items()}
        self.criminal_amount_distributions = {good: self.get_truncated_normal(*tuple(statlist)
                                                                             ) for good, statlist in
                                             self.parameters['criminal_amounts'].items()}

        #this stage_list facilitiates ten different time periods within a day for trades
        stage_list = ['step',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners'
                      ]

        self.schedule = StagedActivation(self, stage_list=stage_list, shuffle=True, shuffle_between_stages=True)
        self.original_suppliers = set()

        self.output_stats = self.output_stats(self.config)
        # Create agents
        agent_count = 0

        if self.parameters['deterministic_mode']:
            num_criminals = math.ceil(self.parameters['num_users'] * self.parameters['chance_of_criminal'])

            # nsuppliers = {good:int(self.parameters['num_users']*chance
            #                       ) for good, chance in self.parameters['chance_of_supplying'].items()}

            # First get the number of suppliers that is to be had in the scenario, by adding up all of the
            # chances of being a supplier , and then taking the percent of the total, flooring with an int again
            # then create a dict for how many of each supplier there are.  then, go through the agents designated
            # as good and assing suppliers, starting from the highest likelihood down to the lowest
            # these are for the good agents. The bad agents go by another algorithm of what they supply, that is
            # related to the price



            #criminal suppliers
            chance_of_supplier = 0
            for good, chance in self.parameters['criminal_chance_of_supplying'].items():
                chance_of_supplier += chance

            num_suppliers1 = int(round(num_criminals * chance_of_supplier))
            sorted_suppliers = sorted(self.parameters['criminal_chance_of_supplying'].items(), key=lambda x: x[1], reverse=True)

            sup_count = 0
            nsuppliers = OrderedDict()
            for good,chance in sorted_suppliers:
                if sup_count < num_suppliers1:
                    normalized = chance/chance_of_supplier
                    rounded = int(round(num_suppliers1 * normalized))
                    num_sup_this_good =  rounded if rounded > 0 else 1
                    num_sup_this_good = min (num_sup_this_good,(num_suppliers1-sup_count))
                    sup_count = sup_count + num_sup_this_good
                    nsuppliers [good]= num_sup_this_good

            agent_supplymap = {}
            current_agent = 0
            for good, num_suppliers in nsuppliers.items():
                for _ in range(num_suppliers):
                    if current_agent not in agent_supplymap:
                        agent_supplymap[current_agent]= []
                    agent_supplymap[current_agent].append(good)
                    next = current_agent +1
                    current_agent  = next if next < num_suppliers1 else 0

            self.agents = {}
            for agent_num,supply_list in agent_supplymap.items():
            #for good, num_suppliers in nsuppliers.items():
                #for _ in range(num_suppliers):
                #agent_str = str(agent_count) + "-0"
                #a = globals()['ReputationAgent'](agent_str, self, criminal=True, supply_list=[good])
                agent_str = str(agent_num) + "-0"
                a = globals()['ReputationAgent'](agent_str, self, criminal=True, supply_list=supply_list)
                self.schedule.add(a)
                self.agents[agent_num]=a
                for supply_item in supply_list:
                    self.criminal_suppliers[supply_item].add(agent_str)
                self.original_suppliers.add(agent_str)
                self.orig[agent_str]= agent_str
                self.m[agent_str]= agent_num
                agent_count += 1



            #criminal consumers
            self.num_criminal_consumers = num_criminals - num_suppliers1
            for _ in range(self.num_criminal_consumers):
                agent_str = str(agent_count) + "-0"
                a = globals()['ReputationAgent'](agent_str, self, criminal=True, supply_list=[])
                self.schedule.add(a)
                self.agents[agent_count]=a
                self.original_suppliers.add(agent_str)
                self.orig[agent_str]= agent_str
                self.m[agent_str]= agent_count
                agent_count += 1

            #good suppliers
            chance_of_supplier = 0
            for good, chance in self.parameters['chance_of_supplying'].items():
                chance_of_supplier += chance
            first_good_supplier_num = agent_count
            num_suppliers1 = int(round((self.parameters['num_users'] -num_criminals) * chance_of_supplier))
            sorted_suppliers = sorted(self.parameters['chance_of_supplying'].items(), key=lambda x: x[1], reverse=True)

            sup_count = 0
            nsuppliers = OrderedDict()
            for good,chance in sorted_suppliers:
                if sup_count < num_suppliers1:
                    normalized = chance/chance_of_supplier
                    rounded = int(round(num_suppliers1 * normalized))
                    num_sup_this_good =  rounded if rounded > 0 else 1
                    num_sup_this_good = min (num_sup_this_good,(num_suppliers1-sup_count))
                    sup_count = sup_count + num_sup_this_good
                    nsuppliers [good]= num_sup_this_good

            agent_supplymap = {}
            current_agent = first_good_supplier_num
            for good, num_suppliers in nsuppliers.items():
                for _ in range(num_suppliers):
                    if current_agent not in agent_supplymap:
                        agent_supplymap[current_agent] = []
                    agent_supplymap[current_agent].append(good)
                    next = current_agent + 1
                    current_agent = next if next < (first_good_supplier_num + num_suppliers1) else first_good_supplier_num


            for agent_num, supply_list in agent_supplymap.items():
                # for good, num_suppliers in nsuppliers.items():
                # for _ in range(num_suppliers):
                # agent_str = str(agent_count) + "-0"
                # a = globals()['ReputationAgent'](agent_str, self, criminal=True, supply_list=[good])
                agent_str = str(agent_num) + "-0"
                a = globals()['ReputationAgent'](agent_str, self, criminal=False, supply_list=supply_list)
                self.schedule.add(a)
                self.agents[agent_num] = a
                self.original_suppliers.add(agent_str)
                self.orig[agent_str] = agent_str
                self.m[agent_str] = agent_num
                agent_count += 1

            # for good, num_suppliers in nsuppliers.items():
            #     for _ in range(num_suppliers):
            #         agent_str = str(agent_count) + "-0"
            #         a = globals()['ReputationAgent'](agent_str, self, criminal=False, supply_list=[good])
            #         self.schedule.add(a)
            #         self.agents[agent_count]=a
            #         self.orig[agent_str]= agent_str
            #         self.m[agent_str]= agent_count
            #         agent_count += 1

            #good consumers
            self.num_good_consumers = self.parameters['num_users'] - agent_count
            for i in range(agent_count,self.parameters['num_users']):
                agent_str = str(agent_count) + "-0"
                a = globals()['ReputationAgent'](agent_str, self,criminal=False, supply_list=[])
                self.schedule.add(a)
                self.agents[agent_count]=a
                self.orig[agent_str]= agent_str
                self.m[agent_str]= agent_count
                agent_count += 1

        else:
            for _ in range(self.parameters['num_users']):
                agent_str = str(agent_count) + "-0"
                a = globals()['ReputationAgent'](agent_str, self)
                self.schedule.add(a)
                self.orig[agent_str]= agent_str
                self.m[agent_str]= agent_count
                agent_count += 1

        self.print_agent_goodness()

        self.good2good_agent_cumul_completed_transactions  = 0
        self.good2good_agent_cumul_total_price = 0
        self.bad2good_agent_cumul_completed_transactions = 0
        self.bad2good_agent_cumul_total_price = 0

        self.good2bad_agent_cumul_completed_transactions  = 0
        self.good2bad_agent_cumul_total_price = 0
        self.bad2bad_agent_cumul_completed_transactions = 0
        self.bad2bad_agent_cumul_total_price = 0

        self.ranks = {}
        if not self.parameters['observer_mode']:
            self.reset_reputation_system()
        self.rank_history = self.rank_history()
        self.reset_stats()

        self.reset_stats_product()

        self.topNPercent = {}
        self.bottomNPercent = {}
        self.anomaly_net = None
        self.predictiveness_net = None
        self.hopfield_net = None
        self.conformity_score = {}
        self.predictiveness_score = {}
        self.hopfield_score = {}
        self.product_labels = []
        self.rater_labels = []
        self.agent_labels = []
        self.reputation_mech = None
        self.hopfield_size = 0

    #print ('Last line of ReputationSim __init__')

    def output_stats(self, config):
        path = config['parameters']['output_path'] + 'output_stats.tsv'

        if not os.path.exists(config['parameters']['output_path']):
            os.makedirs(config['parameters']['output_path'])
        file = open(path, 'a')
        #file.write("hellow world")
        #file.close()

        return file

    def next_product_id(self):
        product_id = self.current_product_id
        self.current_product_id += 1
        return product_id

    def roll_bayes(self,evidence,net):
        #result = self.model.roll_bayes(evidence, "supplier_switch")
        description = self.bayesian_network[net].predict_proba(evidence)
        description_last_entry = len(description) -1
        result = (json.loads(description[description_last_entry].to_json()))['parameters'][0]
        winner = None
        roll = random.uniform(0, 1)
        cumul = 0
        for key, rating in result.items():
            if winner is None:
                cumul = cumul + rating
                if cumul > roll:
                    winner = key
        return winner

    def get_bayesian_networks(self):
        bayesian_networks ={}
        for netname, netinfo in self.config['bayesian_networks'].items():
            dists ={}
            states={}
            edge_list = []
            for distname, distinfo in netinfo['discrete_distributions'].items():
                distinfodict = {k:v for k,v in distinfo.items()}
                dist = DiscreteDistribution(distinfodict)
                dists[distname]=dist
                state = State(dist, name=distname)
                states[distname]= state
            cpts = copy.deepcopy(netinfo['conditional_probability'])
            isa_dag = True
            while len(cpts)> 0 and isa_dag:
                len_before = len(cpts)
                states_exist = {k:v for k,v in cpts.items() if all(name in states for name in v['RVs'])}
                for cptname, cptinfo in states_exist.items():
                    dist_list = [dists[rvname] for rvname in cptinfo['RVs']]
                    cpt = ConditionalProbabilityTable(cptinfo['CPT'],dist_list)
                    dists[cptname ]= cpt
                    state = State(cpt,name=cptname)
                    states[cptname] = state
                    cpts.pop(cptname)
                    for rvname in cptinfo['RVs']:
                        edge_list.append((states[rvname],states[cptname]))
                if len_before == len(cpts):
                    isa_dag = False

            bayesian_networks[netname]= BayesianNetwork(netname)
            state_list = [state for statename,state in states.items()]
            bayesian_networks[netname].add_states(*state_list)
            for edge in edge_list:
                bayesian_networks[netname].add_edge(*edge)
            bayesian_networks[netname].bake()

        #     description = bayesian_networks[netname].predict_proba({})
        #
        #     print ("description = bayesian_networks[netname].predict_proba({})")
        #     print(description)
        #
        #
        # description = bayesian_networks["supplier_scam"].predict_proba(
        #     {"supplier_outside_reputation":"supplier_outside_reputationHigh"})
        # print("""description = bayesian_networks["supplier_scam"].predict_proba(
        #     {"supplier_outside_reputation":"supplier_outside_reputationHigh"})""")
        # print(description)
        #
        # description = bayesian_networks["supplier_scam"].predict_proba(
        #     {"supplier_LDprofit":"supplier_LDprofitNot",
        #     "supplier_outside_reputation":"supplier_outside_reputationLow"})
        # print("""description = bayesian_networks["supplier_scam"].predict_proba(
        #     {"supplier_LDprofit":"supplier_LDprofitNot",
        #     "supplier_outside_reputation":"supplier_outside_reputationLow"})""")
        # print(description)
        #
        # description = bayesian_networks["product_scam"].predict_proba({"product_fairness":"product_fairnessNot"})
        # print('description = bayesian_networks["product_scam"].predict_proba({"product_fairness":"product_fairnessNot"})')
        # print(description)
        #
        # description = bayesian_networks["product_scam"].predict_proba({"product_fairness":"product_fairness"})
        # print('description = bayesian_networks["product_scam"].predict_proba({"product_fairness":"product_fairness"})')
        # print(description)
        #
        # description = bayesian_networks["PLRo"].predict_proba({"PLRt":"PLRt_30","PLRo":"PLRoNot"})
        # print("""description = bayesian_networks["PLRo"].predict_proba({"PLRt":"PLRt_30","PLRo":"PLRoNot"})""")
        # print(description)
        #
        # description = bayesian_networks["num_bought_scams"].predict_proba({"product_Dscam":"product_Dscam_lower"})
        # print('description = bayesian_networks["num_bought_scams"].predict_proba({"product_Dscam":"product_Dscam_lower"})')
        # print(description)
        #

        return bayesian_networks




    def reset_reputation_system(self):
        if self.reputation_system:

            self.reputation_system.clear_ratings()
            self.reputation_system.clear_ranks()
            #self.reputation_system.set_parameters({'fullnorm': True})


    def reset_stats_product(self):

        # OQ is organic quality 0-1
        # CR is commission rate
        #
        # bsl_num = Σorganicbuys(Price * (1 - OQ))
        # bsl_denom = Σorganicbuys(Price)
        # sgp_denom = Σsponsoredbuys(Price * (1 + CR))
        # sgl_num = (Σsponsoredbuys(Price * (1 + CR) * OQ))
        # sgl_denom = (Σorganicbuys(Price * OQ))
        #
        self.bsl_num = 0
        self.sgp2_num = 0
        self.bsl_denom = 0
        self.sgp_denom = 0
        self.sgl_num_num = 0
        self.sgl_denom_num = 0
        self.sgl_num_denom = 0
        self.sgl_denom_denom = 0
        self.num_name_changes = 0
        self.num_decisions_to_scam = 0
        self.num_bought_reviews = 0
        self.num_products_switched = 0
        self.sum_good_consumer_ratings = 0
        self.num_good_consumer_rated_purchases = 0

        self.bsl_num_daily = 0
        self.sgp2_num_daily = 0
        self.bsl_denom_daily = 0
        self.sgp_denom_daily = 0
        self.sgl_num_num_daily = 0
        self.sgl_denom_num_daily = 0
        self.sgl_num_denom_daily = 0
        self.sgl_denom_denom_daily = 0
        self.num_name_changes_daily = 0
        self.num_decisions_to_scam_daily = 0
        self.num_bought_reviews_daily = 0
        self.num_products_switched_daily = 0
        self.sum_good_consumer_ratings_daily = 0
        self.num_good_consumer_rated_purchases_daily = 0

        self.bsl_num_window = []
        self.sgp2_num_window = []
        self.bsl_denom_window = []
        self.sgp_denom_window = []
        self.sgl_num_num_window = []
        self.sgl_denom_num_window = []
        self.sgl_num_denom_window = []
        self.sgl_denom_denom_window = []
        self.num_name_changes_window = []
        self.num_decisions_to_scam_window = []
        self.num_bought_reviews_window  = []
        self.num_products_switched_window = []
        self.sum_good_consumer_ratings_window = []
        self.num_good_consumer_rated_purchases_window = []
        
        self.change_window_day(self.bsl_num_window)
        self.change_window_day(self.sgp2_num_window)
        self.change_window_day(self.bsl_denom_window)
        self.change_window_day(self.sgp_denom_window )
        self.change_window_day(self.sgl_num_num_window )
        self.change_window_day(self.sgl_denom_num_window )
        self.change_window_day(self.sgl_num_denom_window)
        self.change_window_day(self.sgl_denom_denom_window )
        self.change_window_day(self.num_name_changes_window)
        self.change_window_day(self.num_decisions_to_scam_window )
        self.change_window_day(self.num_bought_reviews_window)
        self.change_window_day(self.num_products_switched_window )
        self.change_window_day(self.sum_good_consumer_ratings_window )
        self.change_window_day(self.num_good_consumer_rated_purchases_window )


    def daily_stats_reset(self):

        self.bsl_num_daily = 0
        self.sgp2_num_daily = 0
        self.bsl_denom_daily = 0
        self.sgp_denom_daily = 0
        self.sgl_num_num_daily = 0
        self.sgl_denom_num_daily = 0
        self.sgl_num_denom_daily = 0
        self.sgl_denom_denom_daily = 0
        self.num_name_changes_daily = 0
        self.num_decisions_to_scam_daily = 0
        self.num_bought_reviews_daily = 0
        self.num_products_switched_daily = 0
        self.sum_good_consumer_ratings_daily = 0
        self.num_good_consumer_rated_purchases_daily = 0

        self.change_window_day(self.bsl_num_window)
        self.change_window_day(self.sgp2_num_window)
        self.change_window_day(self.bsl_denom_window)
        self.change_window_day(self.sgp_denom_window )
        self.change_window_day(self.sgl_num_num_window )
        self.change_window_day(self.sgl_denom_num_window )
        self.change_window_day(self.sgl_num_denom_window)
        self.change_window_day(self.sgl_denom_denom_window )
        self.change_window_day(self.num_name_changes_window)
        self.change_window_day(self.num_decisions_to_scam_window)
        self.change_window_day(self.num_bought_reviews_window)
        self.change_window_day(self.num_products_switched_window)
        self.change_window_day(self.sum_good_consumer_ratings_window )
        self.change_window_day(self.num_good_consumer_rated_purchases_window )

    def window_add(self,window,item):
        #add the item to the present day of the window
        window[-1].append(item)

    def change_window_day(self,window):
        if len (window) >= self.parameters["statistics_window_size"] :
            window.pop(0)
        window.append([])

    def window_sum(self, window):
        flat_list = [item for sublist in window for item in sublist]
        return sum(flat_list)

    def add_good_consumer_rating(self, rating):
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.sum_good_consumer_ratings += rating
            self.num_good_consumer_rated_purchases += 1
            self.sum_good_consumer_ratings_daily += rating
            self.num_good_consumer_rated_purchases_daily += 1
            self.window_add(self.sum_good_consumer_ratings_window, rating)
            self.window_add(self.num_good_consumer_rated_purchases_window,1)

    def add_organic_buy(self, price, quality, black_market):
        # bsl_num = Σorganicbuys(Price * (1 - OQ))
        # bsl_denom = Σorganicbuys(Price)
        # sgl_denom_num = (Σorganicbuys(Price * OQ))
        # sgl_denom_denom =  Σorganicbuys(OQ)
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.bsl_num += price * (1.0 - quality)
            if black_market:
                self.sgp2_num += price * (1.0 - quality)
            self.bsl_denom += price
            self.sgl_denom_num += price * quality
            self.sgl_denom_denom += quality
            self.bsl_num_daily += price * (1.0 - quality)
            if black_market:
                self.sgp2_num_daily += price * (1.0 - quality)
            self.bsl_denom_daily += price
            self.sgl_denom_num_daily += price * quality
            self.sgl_denom_denom_daily += quality
            self.window_add(self.bsl_num_window,price * (1.0 - quality))
            if black_market:
                self.window_add(self.sgp2_num_window, price * (1.0 - quality))
            self.window_add(self.bsl_denom_window, price)
            self.window_add(self.sgl_denom_num_window, price * quality)
            self.window_add(self.sgl_denom_denom_window, quality)

    def add_legit_transaction(self, supplier, price):
        if self.daynum >= self.parameters["statistics_initialization"]:
            real_supplier = self.orig[supplier]
            if not real_supplier in self.individual_agent_market_volume:
                self.individual_agent_market_volume[real_supplier] = 0
            self.individual_agent_market_volume[real_supplier] += price

    def add_sponsored_buy(self,price, quality, commission):
        # sgp_denom = Σsponsoredbuys(Price * (1 + CR))
        # sgl_num_num = (Σsponsoredbuys(Price * (1 + CR) * OQ))
        # sgl_num_denom = Σsponsoredbuys(OQ)
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.sgp_denom += commission
            self.sgl_num_num += commission * quality
            self.sgl_num_denom += quality
            self.sgp_denom_daily += commission
            self.sgl_num_num_daily += commission * quality
            self.sgl_num_denom_daily += quality
            self.window_add(self.sgp_denom_window, commission)
            self.window_add(self.sgl_num_num_window, commission * quality)
            self.window_add(self.sgl_num_denom_window ,quality)

    def add_identity_change(self):
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.num_name_changes += 1
            self.num_name_changes_daily += 1
            self.window_add(self.num_name_changes_window, 1)

    def add_decision_to_scam(self):
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.num_decisions_to_scam += 1
            self.num_decisions_to_scam_daily += 1
            self.window_add(self.num_decisions_to_scam_window, 1)

    def add_bought_review(self):
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.num_bought_reviews += 1
            self.num_bought_reviews_daily += 1
            self.window_add(self.num_bought_reviews_window, 1)

    def add_product_switch(self):
        if self.daynum >= self.parameters["statistics_initialization"]:
            self.num_products_switched += 1
            self.num_products_switched_daily += 1
            self.window_add(self.num_products_switched_window, 1)

    def bsl_daily(self):
        bsl = self.bsl_num_daily / self.bsl_denom_daily if self.bsl_denom_daily != 0 else -1
        return bsl

    def sgp2_daily(self):
        sgp = self.sgp2_num_daily / self.sgp_denom_daily if self.sgp_denom_daily != 0 else -1
        return sgp

    def sgp_daily(self):
        sgp = self.bsl_num_daily / self.sgp_denom_daily if self.sgp_denom_daily != 0 else -1
        return sgp

    def sgl_daily(self):
        sgl_num_daily = self.sgl_num_num_daily / self.sgl_num_denom_daily if self.sgl_num_denom_daily != 0 else -1
        sgl_denom_daily = self.sgl_denom_num_daily / self.sgl_denom_denom_daily if self.sgl_denom_denom_daily != 0 else 0
        sgl = sgl_num_daily / sgl_denom_daily if sgl_num_daily != -1 and sgl_denom_daily != 0 else -1
        return sgl

    def asp_daily(self):
        criminals = [bad for good, bad in self.criminal_suppliers.items()]
        asp = (len(criminals) * self.get_end_tick()) / self.num_name_changes_daily if self.num_name_changes_daily != 0 else -1
        return asp

    def utility_daily(self):
        utility = (self.sum_good_consumer_ratings_daily /
                   self.num_good_consumer_rated_purchases_daily if self.num_good_consumer_rated_purchases_daily > 0 else -1)
        return utility

    def bsl_window(self):
        denom_sum = self.window_sum(self.bsl_denom_window)
        bsl = self.window_sum(self.bsl_num_window) /denom_sum  if denom_sum != 0 else -1
        return bsl

    def sgp2_window(self):
        denom_sum = self.window_sum(self.sgp_denom_window)
        sgp = self.window_sum(self.sgp2_num_window) / denom_sum if denom_sum != 0 else -1
        return sgp

    def sgp_window(self):
        denom_sum = self.window_sum(self.sgp_denom_window)
        sgp = self.window_sum(self.bsl_num_window )/ denom_sum if denom_sum != 0 else -1
        return sgp

    def sgl_window(self):
        num_denom_sum = self.window_sum(self.sgl_num_denom_window)
        denom_denom_sum = self.window_sum(self.sgl_denom_denom_window)
        sgl_num = self.window_sum(self.sgl_num_num_window) / num_denom_sum if num_denom_sum != 0 else -1
        sgl_denom = self.window_sum(self.sgl_denom_num_window) / denom_denom_sum if denom_denom_sum != 0 else 0
        sgl = sgl_num / sgl_denom if sgl_num != -1 and sgl_denom != 0 else -1
        return sgl

    def asp_window(self):
        denom_sum = self.window_sum(self.num_name_changes_window)
        criminals = [bad for good, bad in self.criminal_suppliers.items()]
        asp = (len(criminals) * self.get_end_tick()) / denom_sum if denom_sum != 0 else -1
        return asp


    def utility_window(self):
        denom_sum = self.window_sum(self.num_good_consumer_rated_purchases_window)
        utility = (self.window_sum(self.sum_good_consumer_ratings_window )/denom_sum if denom_sum > 0 else -1)
        return utility

    def bsl(self):
        bsl = self.bsl_num / self.bsl_denom if self.bsl_denom != 0 else -1
        return bsl

    def sgp2(self):
        sgp = self.sgp2_num / self.sgp_denom if self.sgp_denom != 0 else -1
        return sgp

    def sgp(self):
        sgp = self.bsl_num / self.sgp_denom if self.sgp_denom != 0 else -1
        return sgp

    def sgl(self):
        sgl_num = self.sgl_num_num / self.sgl_num_denom if self.sgl_num_denom != 0 else -1
        sgl_denom = self.sgl_denom_num / self.sgl_denom_denom if self.sgl_denom_denom != 0 else 0
        sgl = sgl_num / sgl_denom if sgl_num != -1 and sgl_denom != 0 else -1
        return sgl

    def asp(self):
        criminals = [bad for good, bad in self.criminal_suppliers.items()]
        asp = (len(criminals) * self.get_end_tick()) / self.num_name_changes if self.num_name_changes != 0 else -1
        return asp



    def market_volume(self):
        market_volume = ( self.good2good_agent_cumul_total_price +
        self.bad2good_agent_cumul_total_price +
        self.good2bad_agent_cumul_total_price +
        self.bad2bad_agent_cumul_total_price  )
        return market_volume

    def maxproduct(self):
        maxproduct = self.current_product_id
        return maxproduct

    def loss_to_scam(self):
        good2all = (self.good2good_agent_cumul_total_price
                    + self.good2bad_agent_cumul_total_price)
        loss2scam = self.good2bad_agent_cumul_total_price / good2all if good2all > 0 else -1
        return loss2scam

    def profit_from_scam(self):
        # dont include bad2good because that can be thought of as living expenses
        profit_from_scam = (self.good2bad_agent_cumul_total_price /
                            self.bad2bad_agent_cumul_total_price if self.bad2bad_agent_cumul_total_price > 0 else -1)

        return profit_from_scam

    def omut(self):
        all2good = (self.good2good_agent_cumul_total_price + self.bad2good_agent_cumul_total_price)
        organic_market_volume = all2good + self.good2bad_agent_cumul_total_price
        num_consumers = self.num_good_consumers + self.num_criminal_consumers
        good_consumer_fraction = self.num_good_consumers/num_consumers if num_consumers > 0 else 0
        possible_market_volume = organic_market_volume * good_consumer_fraction
        omut = self.good2good_agent_cumul_total_price/possible_market_volume if possible_market_volume > 0 else -1
        return omut

    def utility(self):
        utility = (self.sum_good_consumer_ratings/
                   self.num_good_consumer_rated_purchases if self.num_good_consumer_rated_purchases > 0 else -1)
        return utility

    def goodness(self,agentstr):
        agentnum = self.m[self.orig[agentstr]]
        agent =  self.agents[agentnum]
        goodness = agent.goodness if agent is not None else -1
        return goodness

    def inequity(self):

        equitable_shares = [
            self.individual_agent_market_volume[agent]/self.goodness(agent)
            if agent in self.individual_agent_market_volume and self.goodness(agent) > 0
            else self.individual_agent_market_volume[agent] / 0.0001
            if agent in self.individual_agent_market_volume else 0
            for agent in self.original_suppliers]
        sorted_shares = sorted(equitable_shares)
        N = len(sorted_shares)
        if N and sum(sorted_shares):
            B = sum(xi * (N - i) for i, xi in enumerate(sorted_shares)) / (N * sum(sorted_shares))
            inequity = (1 + (1 / N) - 2 * B)
        else:
            inequity = sys.maxsize

        return inequity

    def print_stats_product(self):

        bsl_daily = self.bsl_daily()
        if bsl_daily is None or not isinstance(bsl_daily, float):
            bsl_daily = -1.00
        sgp_daily = self.sgp_daily()
        if sgp_daily is None or not isinstance(sgp_daily, float):
            sgp_daily = -1.00
        sgp2_daily = self.sgp2_daily()
        if sgp2_daily is None or not isinstance(sgp2_daily, float):
            sgp2_daily = -1.00
        sgl_daily = self.sgl_daily()
        if sgl_daily is None or not isinstance(sgl_daily, float):
            sgl_daily = -1.00
        asp_daily = self.asp_daily()
        if asp_daily is None or not isinstance(asp_daily, float):
            asp_daily = -1.00
        num_name_changes_daily = self.num_name_changes_daily
        #if num_name_changes_daily is None or not isinstance(num_name_changes_daily, float):
        #    num_name_changes_daily = -1.00
        num_decisions_to_scam_daily = self.num_decisions_to_scam_daily
        #if num_decisions_to_scam_daily is None or not isinstance(num_decisions_to_scam_daily, float):
        #    num_decisions_to_scam_daily = -1.00
        num_bought_reviews_daily = self.num_bought_reviews_daily
        #if num_bought_reviews_daily is None or not isinstance(num_bought_reviews_daily, float):
        #    num_bought_reviews_daily = -1.00
        num_products_switched_daily = self.num_products_switched_daily
        #if num_products_switched_daily is None or not isinstance(num_products_switched_daily, float):
        #    num_products_switched_daily = -1.00

        utility_daily = self.utility_daily()
        if utility_daily is None or not isinstance(utility_daily, float):
            utility_daily = -1.00

        bsl_window = self.bsl_window()
        if bsl_window is None or not isinstance(bsl_window, float):
            bsl_window = -1.00
        sgp_window = self.sgp_window()
        if sgp_window is None or not isinstance(sgp_window, float):
            sgp_window = -1.00
        sgp2_window = self.sgp2_window()
        if sgp2_window is None or not isinstance(sgp2_window, float):
            sgp2_window = -1.00
        sgl_window = self.sgl_window()
        if sgl_window is None or not isinstance(sgl_window, float):
            sgl_window = -1.00
        asp_window = self.asp_window()
        if asp_window is None or not isinstance(asp_window, float):
            asp_window = -1.00
        num_name_changes_window = self.window_sum(self.num_name_changes_window)
        #if num_name_changes_window is None or not isinstance(num_name_changes_window, float):
        #    num_name_changes_window = -1.00
        num_decisions_to_scam_window = self.window_sum(self.num_decisions_to_scam_window)
        #if num_decisions_to_scam_window is None or not isinstance(num_decisions_to_scam_window, float):
        #    num_decisions_to_scam_window = -1.00
        num_bought_reviews_window = self.window_sum(self.num_bought_reviews_window)
        #if num_bought_reviews_window is None or not isinstance(num_bought_reviews_window, float):
        #    num_bought_reviews_window = -1.00
        num_products_switched_window = self.window_sum(self.num_products_switched_window)
        #if num_products_switched_window is None or not isinstance(num_products_switched_window, float):
        #    num_products_switched_window = -1.00

        utility_window = self.utility_window()
        if utility_window is None or not isinstance(utility_window, float):
            utility_window = -1.00

        bsl = self.bsl()
        if bsl is None or not isinstance(bsl, float):
            bsl = -1.00
        sgp = self.sgp()
        if sgp is None or not isinstance(sgp, float):
            sgp = -1.00
        sgp2 = self.sgp2()
        if sgp2 is None or not isinstance(sgp2, float):
            sgp2 = -1.00
        sgl = self.sgl()
        if sgl is None or not isinstance(sgl, float):
            sgl = -1.00
        asp = self.asp()
        if asp is None or not isinstance(asp, float):
            asp = -1.00
        num_name_changes = self.num_name_changes
        #if num_name_changes is None or not isinstance(num_name_changes, float):
        #    num_name_changes = -1.00
        num_decisions_to_scam = self.num_decisions_to_scam
        #if num_decisions_to_scam is None or not isinstance(num_decisions_to_scam, float):
        #    num_decisions_to_scam = -1.00
        num_bought_reviews = self.num_bought_reviews
        #if num_bought_reviews is None or not isinstance(num_bought_reviews, float):
        #    num_bought_reviews = -1.00
        num_products_switched = self.num_products_switched
        #if num_products_switched is None or not isinstance(num_products_switched, float):
        #    num_products_switched = -1.00
        omut = self.omut()
        if omut is None or not isinstance(omut, float):
            omut = -1.00
        market_volume = self.market_volume()
        if market_volume is None or not isinstance(market_volume, float):
            market_volume = -1.00
        maxproduct = self.maxproduct()
        #if maxproduct is None or not isinstance(maxproduct, float):
        #    maxproduct = -1.00
        lts = self.loss_to_scam()
        if lts is None or not isinstance(lts, float):
            lts = -1.00
        pfs = self.profit_from_scam()
        if pfs is None or not isinstance(pfs, float):
            pfs = -1.00
        utility = self.utility()
        if utility is None or not isinstance(utility, float):
            utility = -1.00
        inequity = self.inequity()
        if inequity is None or not isinstance(inequity, float):
            inequity = -1.00

        if self.daynum % 30 == 0 and self.daynum >= self.parameters["statistics_initialization"]:
            print(
                """\n time:{11}, bsl:{0:.4f}, sgp:{1:.4f},  sgp2:{12:.4f}, sgl:{2:.4f}, asp:{3:.4f}, utility:{9:.4},num_name_changes:{25:.4f},num_decisions_to_scam:{26:.4f} ,num_bought_reviews:{27:.4f},num_products_switched:{28:.4f},bsl_daily:{13:.4f}, sgp_daily:{14:.4f},sgp2_daily:{15:.4f}, sgl_daily:{16:.4f}, asp_daily:{17:.4f}, utility_daily:{18:.4},num_name_changes_daily:{29:.4f},num_decisions_to_scam_daily:{30:.4f} ,num_bought_reviews_daily:{31:.4f},num_products_switched_daily:{32:.4f},bsl_window:{19:.4f}, sgp_window:{20:.4f},  sgp2_window:{21:.4f}, sgl_window:{22:.4f}, asp_window:{23:.4f}, utility_window:{24:.4},num_name_changes_window:{33:.4f},num_decisions_to_scam_window:{34:.4f} ,num_bought_reviews_window:{35:.4f},num_products_switched_window:{36:.4f} omut:{4:.4f}, market volume:{5:.4f}, maxproduct:{6:.4f}, lts:{7:.4f}, pfs:{8:.4f}, inequity:{10:.4f}\n""".format(
                    bsl, sgp, sgl, asp, omut, market_volume, maxproduct, lts, pfs, utility, inequity, self.daynum,
                    sgp2,
                    bsl_daily, sgp_daily, sgp2_daily, sgl_daily, asp_daily, utility_daily,
                    bsl_window, sgp_window, sgp2_window, sgl_window, asp_window, utility_window,
                    num_name_changes, num_decisions_to_scam, num_bought_reviews, num_products_switched,
                    num_name_changes_daily, num_decisions_to_scam_daily, num_bought_reviews_daily,
                    num_products_switched_daily,
                    num_name_changes_window, num_decisions_to_scam_window, num_bought_reviews_window,
                    num_products_switched_window
                ))
        if self.error_log and self.daynum >= self.parameters["statistics_initialization"]:
            self.error_log.write(
                """\n time:{11}, bsl:{0:.4f}, sgp:{1:.4f},  sgp2:{12:.4f}, sgl:{2:.4f}, asp:{3:.4f}, utility:{9:.4},num_name_changes:{25:.4f},num_decisions_to_scam:{26:.4f} ,num_bought_reviews:{27:.4f},num_products_switched:{28:.4f},bsl_daily:{13:.4f}, sgp_daily:{14:.4f},sgp2_daily:{15:.4f}, sgl_daily:{16:.4f}, asp_daily:{17:.4f}, utility_daily:{18:.4},num_name_changes_daily:{29:.4f},num_decisions_to_scam_daily:{30:.4f} ,num_bought_reviews_daily:{31:.4f},num_products_switched_daily:{32:.4f},bsl_window:{19:.4f}, sgp_window:{20:.4f},  sgp2_window:{21:.4f}, sgl_window:{22:.4f}, asp_window:{23:.4f}, utility_window:{24:.4},,num_name_changes_window:{33:.4f},num_decisions_to_scam_window:{34:.4f} ,num_bought_reviews_window:{35:.4f},num_products_switched_window:{36:.4f} omut:{4:.4f}, market volume:{5:.4f}, maxproduct:{6:.4f}, lts:{7:.4f}, pfs:{8:.4f}, inequity:{10:.4f}\n""".format(
                    bsl, sgp, sgl, asp, omut, market_volume, maxproduct, lts, pfs, utility, inequity, self.daynum,
                    sgp2,
                    bsl_daily, sgp_daily, sgp2_daily, sgl_daily, asp_daily, utility_daily,
                    bsl_window, sgp_window, sgp2_window, sgl_window, asp_window, utility_window,
                    num_name_changes, num_decisions_to_scam, num_bought_reviews, num_products_switched,
                    num_name_changes_daily, num_decisions_to_scam_daily, num_bought_reviews_daily,
                    num_products_switched_daily,
                    num_name_changes_window, num_decisions_to_scam_window, num_bought_reviews_window,
                    num_products_switched_window
                ))

    def write_output_stats_line(self):
        line = []
        line.append(self.study_path)
        for new_val in self.config['output_columns']:
            # old_val = configfile['parameters']
            old_val = self.config
            old_old_val = old_val
            while isinstance(new_val, dict) and len(new_val) == 1:
                # while isinstance(new_val, dict):
                nextKey = next(iter(new_val.items()))[0]
                old_old_val = old_val
                old_val = old_val[nextKey]
                new_val = new_val[nextKey]
            # old_old_val[nextKey] = new_val
            paramlist = new_val
            for param in paramlist:
                length = len(old_old_val[nextKey][param]) if isinstance(old_old_val[nextKey][param],
                                                                        list) else 1
                if length > 1:
                    for i in range(length):
                        new_param = "{0}{1}".format(param, i)
                        line.append(old_old_val[nextKey][param][i])
                else:
                    line.append(old_old_val[nextKey][param])
        for test, _ in self.config['tests']['default'].items():
            v = ""
            if test == "OMUT":
                v = self.omut()
            elif test == "maxproduct":
                v = self.maxproduct()
            elif test == "bsl":
                v = self.bsl()
            elif test == "sgp":
                v = self.sgp()
            elif test == "sgp2":
                v = self.sgp2()
            elif test == "sgl":
                v = self.sgl()
            elif test == "asp":
                v = self.asp()
            elif test == "num_name_changes":
                v = self.num_name_changes
            elif test == "num_decisions_to_scam":
                v = self.num_decisions_to_scam
            elif test == "num_bought_reviews":
                v = self.num_bought_reviews
            elif test == "num_products_switched":
                v = self.num_products_switched
            elif test == "utility":
                v = self.utility()
            elif test == "bsl_daily":
                v = self.bsl_daily()
            elif test == "sgp_daily":
                v = self.sgp_daily()
            elif test == "sgp2_daily":
                v = self.sgp2_daily()
            elif test == "sgl_daily":
                v = self.sgl_daily()
            elif test == "asp_daily":
                v = self.asp_daily()
            elif test == "num_name_changes_daily":
                v = self.num_name_changes_daily
            elif test == "num_decisions_to_scam_daily":
                v = self.num_decisions_to_scam_daily
            elif test == "num_bought_reviews_daily":
                v = self.num_bought_reviews_daily
            elif test == "num_products_switched_daily":
                v = self.num_products_switched_daily
            elif test == "utility_daily":
                v = self.utility_daily()
            elif test == "bsl_window":
                v = self.bsl_window()
            elif test == "sgp_window":
                v = self.sgp_window()
            elif test == "sgp2_window":
                v = self.sgp2_window()
            elif test == "sgl_window":
                v = self.sgl_window()
            elif test == "asp_window":
                v = self.asp_window()
            elif test == "num_name_changes_window":
                v = self.window_sum(self.num_name_changes_window)
            elif test == "num_decisions_to_scam_window":
                v = self.window_sum(self.num_decisions_to_scam_window)
            elif test == "num_bought_reviews_window":
                v = self.window_sum(self.num_bought_reviews_window)
            elif test == "num_products_switched_window":
                v = self.window_sum(self.num_products_switched_window)
            elif test == "utility_window":
                v = self.utility_window()
            elif test == "loss_to_scam":
                v = self.loss_to_scam()
            elif test == "profit_from_scam":
                v = self.profit_from_scam()
            elif test == "market_volume":
                v = self.market_volume()
            elif test == "inequity":
                v = self.inequity()
        line.append(v)
        line.append("\n")
        linestr = "\t".join(map(str, line))
        self.output_stats.write(linestr)
        self.output_stats.flush()


    def reset_stats(self):
        self.good2good_agent_completed_transactions  = 0
        self.good2good_agent_total_price = 0

        self.bad2good_agent_completed_transactions = 0
        self.bad2good_agent_total_price = 0

        self.good2bad_agent_completed_transactions  = 0
        self.good2bad_agent_total_price = 0

        self.bad2bad_agent_completed_transactions = 0
        self.bad2bad_agent_total_price = 0

    def get_end_tick(self):
        #final_tick = (final_epoch - initial_epoch) / (days / tick * miliseconds / day)

        secs = self.final_epoch - self.initial_epoch
        final_tick = secs/(self.parameters['days_per_tick'] * self.seconds_per_day)
        return final_tick


    def transaction_report(self):
        #path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] [:-1] + '.tsv'
        file = open(path, "w")
        return(file)

    def error_log(self):
        #path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'errorLog_' +self.parameters['param_str'] [:-1] + '.tsv'
        file = open(path, "w")
        return(file)

    def reset_current_ranks(self):
        for good,supplierlist in self.suppliers.items():
            for supplier in supplierlist:
                self.reset_current_rank(supplier)


    def reset_current_rank(self,id):
        #an alias has first appeared
        self.current_rank_sums[id] = 0
        #self.current_rank_products[id] =0
        self.current_rank_num_adds[id] =0

    def add_rank(self,id,rank):

        self.rank_sums[id] += rank
        self.rank_days[id]  += 1
        self.current_rank_sums[id] += rank
        self.current_rank_num_adds[id]  += 1

    def get_current_avg_rank(self):
        return self.daily_avg_rank

    def calculate_daily_avg_rank(self):
        current_rank_sums = [sum for id, sum in self.current_rank_sums.items()]
        current_rank_num_adds = [sum for id, sum in self.current_rank_num_adds.items()]
        denom = sum(current_rank_num_adds)
        daily_avg_rank = int(round(sum(current_rank_sums)/ denom)) if denom>0 else -1
        return daily_avg_rank

    def get_current_rank(self,id):
        current_rank = int(round(self.current_rank_sums[id]/ self.current_rank_num_adds[id])) if self.current_rank_num_adds[id]>0 else -1
        return current_rank

    def get_current_ranks(self):
        current_ranks = {}
        for good,supplierlist in self.suppliers.items():
            for supplier in supplierlist:
                current_ranks[supplier] = self.get_current_rank(supplier)
        return current_ranks

    def initialize_rank(self,id):
        #an alias has first appeared
        self.rank_sums[id] = 0
        self.rank_days[id] =0


    def get_avg_rank(self,id):
        average_rank = int(round(self.rank_sums[id]/ self.rank_days[id])) if self.rank_days[id]>0 else -1
        return average_rank

    def finalize_rank(self,id):
        if id in self.rank_days and  self.rank_days[id]>0:
            average_rank = self.get_avg_rank(id)
            self.average_rank_history.write("{0}\t{1}\n".format(id,average_rank))

    def finalize_all_ranks(self):
        for good,agentlist in self.suppliers.items():
            for agent in agentlist:
                self.finalize_rank(agent)
        self.average_rank_history.close()


    def average_rank_history(self):
        #path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'averageRankHistory_' +self.parameters['param_str'] [:-1] + '.tsv'
        file = open(path, "w")

        #file.write('time\t')
        heading_list = []
        heading_list.append('time')
        heading_list.append('average_rank')
        heading_list.append('\n')
        average_rank_history_heading = "\t".join(map (str, heading_list))
        file.write(average_rank_history_heading)
        self.rank_sums = {}
        self.rank_days = {}
        self.current_rank_sums = {}
        self.current_rank_num_adds = {}
        return(file)

    def rank_history(self):
        #path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'rankHistory_' +self.parameters['param_str'] [:-1] + '.tsv'
        file = open(path, "w")

        #file.write('time\t')
        heading_list = []
        heading_list.append('time')
        for i in range(len(self.agents)):
            #file.write('{0}\t'.format(self.schedule.agents[i].unique_id))
            heading_list.append(self.agents[i].unique_id)
        #make room for columns to be added on. they need to have headings now so pandas can parse them
        num_extra_agents = int(((self.parameters['num_users'] * self.parameters['chance_of_criminal']*self.end_tick
                            )/self.parameters['scam_parameters']['scam_period'][0]))+1
        for i in range(num_extra_agents+5):
            heading_list.append('alias{0}'.format(i))
        #file.write('\n')
        heading_list.append('\n')
        self.rank_history_heading = "\t".join(map (str, heading_list))
        file.write(self.rank_history_heading)
        return(file)

    def write_rank_history_line(self):
        heading_list = []
        heading_list.append('time')
        #self.rank_history_heading = '{0}\t'.format('time')
        time = int(round(self.schedule.time))
        self.rank_history.write('{0}\t'.format(time))
        key_sort = [self.parse(key)['agent'] for key in self.ranks.keys()]
        key_sort.sort()
        od = OrderedDict()
        for key in key_sort:
            strkey = str(key)
            od[strkey]= self.ranks[strkey]
        #intdir = {int(key): val for key, val in self.ranks.items() }
        #od = OrderedDict(sorted(intdir))
        lastAgent = None
        if len(od):
            for agent,rank in od.items():
                if not agent is None:
                    intagent = int(agent)
                    if lastAgent is None:
                        lastAgent = intagent
                        for i in range (0,intagent):
                            #self.rank_history.write('{0}:{1}\t'.format(i,-1))
                            self.rank_history.write('{0}\t'.format(-1))
                            heading_list.append(i)
                    if intagent < len(self.agents):
                        for i in range (lastAgent+1,intagent):
                            #self.rank_history.write('{0}:{1}\t'.format(i,-1))
                            self.rank_history.write('{0}\t'.format(-1))
                            heading_list.append(i)
                    else:
                        for i in range (lastAgent+1,len(self.agents)):
                            #self.rank_history.write('{0}:{1}\t'.format(i,-1))
                            self.rank_history.write('{0}\t'.format(-1))
                            heading_list.append(i)
                    #self.rank_history.write('{0}:{1}\t'.format(intagent,rank))
                    self.rank_history.write('{0}\t'.format(rank))
                    heading_list.append(intagent)
                    lastAgent = intagent
            for i in range(lastAgent+1,len(self.agents) ):
                #self.rank_history.write('{0}:{1}\t'.format(i,-1))
                self.rank_history.write('{0}\t'.format(-1))
                heading_list.append(i)
            self.rank_history.write('\n')
            heading_list.append('\n')
            self.rank_history_heading = "\t".join(map(str,heading_list))
        else:

            self.rank_history.write('\n')

            # for i in range(len(self.agents)):
            #     id = str(self.schedule.agents[i].unique_id)
            #     rank = od[id] if id in od else -1
            #     self.rank_history.write('{0}\t'.format(rank))
            # self.rank_history.write('\n')

    def write_current_rank_history_line(self):
        pass
        """
                heading_list = []
                heading_list.append('time')
                time = int(round(self.schedule.time))
                self.rank_history.write('{0}\t'.format(time))
                ranks = self.get_current_ranks()
                key_sort = [key for key in ranks.keys()]
                key_sort.sort()
                od = OrderedDict()
                for key in key_sort:
                    strkey = str(key)
                    od[strkey] = ranks[key]
                lastAgent = None
                if len(od):
                    for agent, rank in od.items():
                        if not agent is None:
                            intagent = int(agent)
                            if lastAgent is None:
                                lastAgent = intagent
                                for i in range(0, intagent):
                                    self.rank_history.write('{0}\t'.format(-1))
                                    heading_list.append(i)
                            if intagent < len(self.agents):
                                for i in range(lastAgent + 1, intagent):
                                    self.rank_history.write('{0}\t'.format(-1))
                                    heading_list.append(i)
                            else:
                                for i in range(lastAgent + 1, len(self.agents)):
                                    self.rank_history.write('{0}\t'.format(-1))
                                    heading_list.append(i)
                            self.rank_history.write('{0}\t'.format(rank))
                            heading_list.append(intagent)
                            lastAgent = intagent
                    for i in range(lastAgent + 1, len(self.agents)):
                        self.rank_history.write('{0}\t'.format(-1))
                        heading_list.append(i)
                    self.rank_history.write('\n')
                    heading_list.append('\n')
                    self.rank_history_heading = "\t".join(map(str, heading_list))
                else:

                    self.rank_history.write('\n')"""


    def market_volume_report(self):
        #path = self.parameters['output_path'] + 'transactions_' +self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'marketVolume_' +self.parameters['param_str'] [:-1] + '.tsv'
        file = open(path, "w")

        file.write(
             "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\t{31}\t{32}\n".format(
               "time", "good2bad daily avg price", "good2bad cumul avg price", "good2bad daily avg num transactions",
                "good2bad cumul avg num transactions",
                "good2bad daily avg market vol", "good2bad cumul avg market vol",
                "bad2bad daily avg price", "bad2bad cumul avg price", "bad2bad daily avg num transactions",
                "bad2bad cumul avg num transactions",
                "bad2bad daily avg market vol", "bad2bad cumul avg market vol",
                "good2good daily avg price", "good2good cumul avg price", "good2good daily avg num transactions",
                "good2good cumul avg num transactions",
                "good2good daily avg market vol", "good2good cumul avg market vol",
                "bad2good daily avg price", "bad2good cumul avg price", "bad2good daily avg num transactions",
                "bad2good cumul avg num transactions",
                "bad2good daily avg market vol", "bad2good cumul avg market vol",
                "average price ratio", "latest price ratio", "average num transactions ratio",
                "latest num transactions ratio", "average market volume", "latest market volume",
                "average cost of being bad", "latest cost of being bad"))

        return(file)



    def get_epoch(self, date_time):
        #date_time = '29.08.2011 11:05:02'
        pattern = '%d.%m.%Y %H:%M:%S'
        epoch = int(time.mktime(time.strptime(date_time, pattern)))

        return epoch


    def get_datetime(self, date_time):
        #date_time = '29.08.2011 11:05:02'
        pattern = '%d.%m.%Y %H:%M:%S'
        date_tuple = time.strptime(date_time, pattern)
        date = dt.date(date_tuple[0], date_tuple[1], date_tuple[2])

        return date

    def get_next_transaction(self):
        if not self.transaction_numbers:
            self.transaction_numbers = list( range(0,10000))
            shuffle( self.transaction_numbers)
        self.next_transaction = self.transaction_numbers.pop()
        return self.next_transaction

    def send_trade_to_reputation_system(self, from_agent, to_agent, payment, tags,  payment_unit='', parent = '',rating = '',
                                      type = 'payment'):
        if self.reputation_system is not None:
            value_val = float(rating if rating else payment)
            date = self.since + dt.timedelta(days=self.daynum)
            if rating:
                self.reputation_system.put_ratings([{'from': from_agent, 'type': type, 'to': to_agent,
                                                              'value': value_val,'weight':int(payment), 'time': date}])
                # if self.error_log:
                #     self.error_log.write(str([{'from': from_agent, 'type': type, 'to': to_agent,
                #                                                     'value': value_val,'weight':int(payment), 'time': date}])+ "\n")

            else:
                self.reputation_system.put_ratings([{'from': from_agent, 'type': type, 'to': to_agent,
                                                     'value': value_val, 'time': date}])
                if self.error_log:
                    self.error_log.write(str([{'from': from_agent, 'type': type, 'to': to_agent,
                                           'value': value_val, 'time': date}]) + "\n")

    def print_transaction_report_line(self, from_agent, to_agent, payment, tags,  payment_unit='', parent = '',rating = '',
                                      type = 'payment', scam = False):
        time = (self.schedule.time * self.parameters['days_per_tick']* self.seconds_per_day) + self.initial_epoch
        time = int(time + random.uniform (0,self.seconds_per_day/10))

        network_val = self.parameters['network']
        timestamp_val = time
        type_val = type
        from_val = from_agent
        to_val = to_agent
        value_val = rating if rating else payment
        unit_val = '' if rating else payment_unit
        parent_val = self.get_next_transaction() if rating else parent
        child_val = self.get_next_transaction()
        title_val = ''
        input_val = ''
        tags_val = tags
        format_val = ''
        block_val = ''
        parent_value_val = payment if rating else ''
        parent_unit_val = payment_unit if rating else ''

        self.transaction_report.write(
            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\n".format(
            network_val,timestamp_val,type_val,from_val,to_val,value_val,unit_val,child_val,parent_val,title_val,
            input_val,tags_val,format_val,block_val,parent_value_val,parent_unit_val,scam))

        #self.transaction_report.flush()

    def print_market_volume_report_line(self):
        time = self.schedule.time -1
        good2good_daily_avg_price= self.good2good_agent_total_price/self.good2good_agent_completed_transactions if self.good2good_agent_completed_transactions else 0
        good2good_cumul_avg_price= self.good2good_agent_cumul_total_price/self.good2good_agent_cumul_completed_transactions if self.good2good_agent_cumul_completed_transactions else 0
        good2good_daily_avg_num_transactions=self.good2good_agent_completed_transactions
        good2good_cumul_avg_num_transactions=self.good2good_agent_cumul_completed_transactions
        good2good_daily_avg_market_vol=self.good2good_agent_total_price
        good2good_cumul_avg_market_vol=self.good2good_agent_cumul_total_price
        good2bad_daily_avg_price = self.good2bad_agent_total_price / self.good2bad_agent_completed_transactions if self.good2bad_agent_completed_transactions else 0
        good2bad_cumul_avg_price = self.good2bad_agent_cumul_total_price / self.good2bad_agent_cumul_completed_transactions if self.good2bad_agent_cumul_completed_transactions else 0
        good2bad_daily_avg_num_transactions = self.good2bad_agent_completed_transactions
        good2bad_cumul_avg_num_transactions = self.good2bad_agent_cumul_completed_transactions
        good2bad_daily_avg_market_vol = self.good2bad_agent_total_price
        good2bad_cumul_avg_market_vol = self.good2bad_agent_cumul_total_price
        bad2bad_daily_avg_price = self.bad2bad_agent_total_price / self.bad2bad_agent_completed_transactions if self.bad2bad_agent_completed_transactions else 0
        bad2bad_cumul_avg_price = self.bad2bad_agent_cumul_total_price / self.bad2bad_agent_cumul_completed_transactions if self.bad2bad_agent_cumul_completed_transactions else 0
        bad2bad_daily_avg_num_transactions = self.bad2bad_agent_completed_transactions
        bad2bad_cumul_avg_num_transactions = self.bad2bad_agent_cumul_completed_transactions
        bad2bad_daily_avg_market_vol = self.bad2bad_agent_total_price
        bad2bad_cumul_avg_market_vol = self.bad2bad_agent_cumul_total_price
        bad2good_daily_avg_price = self.bad2good_agent_total_price / self.bad2good_agent_completed_transactions if self.bad2good_agent_completed_transactions else 0
        bad2good_cumul_avg_price = self.bad2good_agent_cumul_total_price / self.bad2good_agent_cumul_completed_transactions if self.bad2good_agent_cumul_completed_transactions else 0
        bad2good_daily_avg_num_transactions = self.bad2good_agent_completed_transactions
        bad2good_cumul_avg_num_transactions = self.bad2good_agent_cumul_completed_transactions
        bad2good_daily_avg_market_vol = self.bad2good_agent_total_price
        bad2good_cumul_avg_market_vol = self.bad2good_agent_cumul_total_price
        avg_price_ratio = (good2good_cumul_avg_price+good2bad_cumul_avg_price)/bad2bad_cumul_avg_price if bad2bad_cumul_avg_price else 0
        latest_price_ratio = (good2good_daily_avg_price+good2bad_daily_avg_price)/bad2bad_daily_avg_price if bad2bad_daily_avg_price else 0
        avg_num_transactions_ratio = (good2good_cumul_avg_num_transactions+good2bad_cumul_avg_num_transactions)/bad2bad_cumul_avg_num_transactions if bad2bad_cumul_avg_num_transactions else 0
        latest_num_transactions_ratio = (good2good_daily_avg_num_transactions+good2bad_daily_avg_num_transactions)/bad2bad_daily_avg_num_transactions if bad2bad_daily_avg_num_transactions else 0
        avg_market_volume = (good2good_cumul_avg_market_vol+good2bad_cumul_avg_market_vol)/bad2bad_cumul_avg_market_vol if bad2bad_cumul_avg_market_vol else 0
        latest_market_volume = (good2good_daily_avg_market_vol+good2bad_daily_avg_market_vol)/bad2bad_daily_avg_market_vol if bad2bad_daily_avg_market_vol else 0
        avg_cost_of_being_bad = bad2bad_cumul_avg_market_vol/good2bad_cumul_avg_market_vol if good2bad_cumul_avg_market_vol else 0
        latest_cost_of_being_bad = bad2bad_daily_avg_market_vol/good2bad_daily_avg_market_vol if good2bad_daily_avg_market_vol else 0


        self.market_volume_report.write(
            "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\t{13}\t{14}\t{15}\t{16}\t{17}\t{18}\t{19}\t{20}\t{21}\t{22}\t{23}\t{24}\t{25}\t{26}\t{27}\t{28}\t{29}\t{30}\t{31}\t{32}\n".format(
            time, good2bad_daily_avg_price,good2bad_cumul_avg_price,good2bad_daily_avg_num_transactions,good2bad_cumul_avg_num_transactions,
            good2bad_daily_avg_market_vol,good2bad_cumul_avg_market_vol,bad2bad_daily_avg_price,bad2bad_cumul_avg_price,
            bad2bad_daily_avg_num_transactions,bad2bad_cumul_avg_num_transactions,bad2bad_daily_avg_market_vol,bad2bad_cumul_avg_market_vol,good2good_daily_avg_price,good2good_cumul_avg_price,good2good_daily_avg_num_transactions,good2good_cumul_avg_num_transactions,
            good2good_daily_avg_market_vol,good2good_cumul_avg_market_vol,bad2good_daily_avg_price,bad2good_cumul_avg_price,
            bad2good_daily_avg_num_transactions,bad2good_cumul_avg_num_transactions,bad2good_daily_avg_market_vol,bad2good_cumul_avg_market_vol,
            avg_price_ratio, latest_price_ratio,avg_num_transactions_ratio, latest_num_transactions_ratio,
            avg_market_volume, latest_market_volume, avg_cost_of_being_bad, latest_cost_of_being_bad))
        self.market_volume_report.flush()
        self.reset_stats()


    def save_info_for_market_volume_report(self, consumer, supplierstr, payment):
            # if increment num transactions and add cum price to the correct agent category of seller
            if self.daynum >= self.parameters["statistics_initialization"]:
                supplier = self.m[supplierstr]
                if self.agents[supplier].good and consumer.good:
                    self.good2good_agent_completed_transactions += 1
                    self.good2good_agent_cumul_completed_transactions += 1
                    self.good2good_agent_total_price += payment
                    self.good2good_agent_cumul_total_price += payment
                elif not self.agents[supplier].good and consumer.good:
                    self.good2bad_agent_completed_transactions += 1
                    self.good2bad_agent_cumul_completed_transactions += 1
                    self.good2bad_agent_total_price += payment
                    self.good2bad_agent_cumul_total_price += payment
                elif not self.agents[supplier].good and not consumer.good:
                    self.bad2bad_agent_completed_transactions += 1
                    self.bad2bad_agent_cumul_completed_transactions += 1
                    self.bad2bad_agent_total_price += payment
                    self.bad2bad_agent_cumul_total_price += payment
                else:  #shouldnt happen
                    self.bad2good_agent_completed_transactions += 1
                    self.bad2good_agent_cumul_completed_transactions += 1
                    self.bad2good_agent_total_price += payment
                    self.bad2good_agent_cumul_total_price += payment


    def print_agent_goodness (self, userlist = [-1]):
        #output a list of given users, sorted by goodness.  if the first item of the list is -1, then output all users

        #path = self.parameters['output_path'] + 'users_' + self.parameters['param_str'] + self.time[0:10] + '.tsv'
        path = self.parameters['output_path'] + 'users_' + self.parameters['param_str'][: -1]  + '.tsv'

        with open(path, 'w') as outfile:
            agents = self.agents if userlist and userlist[0] == -1 else userlist
            outlist = [(agent.unique_id, agent.goodness) for idx,agent in agents.items()]
            sorted_outlist = sorted(outlist,  key=operator.itemgetter(1), reverse=True)
            for id, goodness in sorted_outlist:
                outfile.write("{0}\t{1}\n".format(id, goodness))
        outfile.close()

        path = self.parameters['output_path'] + 'boolean_users_' + self.parameters['param_str'][: -1] + '.tsv'

        with open(path, 'w') as outfile:
            agents = self.agents if userlist and userlist[0] == -1 else userlist
            outlist = [(agent.unique_id, agent.good) for idx, agent in agents.items()]
            sorted_outlist = sorted(outlist, key=operator.itemgetter(1), reverse=True)
            for id, good in sorted_outlist:
                val = 1 if good else 0
                outfile.write("{0}\t{1}\n".format(id, val))
        outfile.close()


    def int_id(self, id):
        root_id_match = self.id_pattern.search(id)
        root_id = root_id_match.group(1)
        return int(root_id)

    def parse(self,agent_string):
        parse = {}
        agent_string_split = agent_string.split(".")
        parse['agentint']= self.int_id(agent_string_split[0])  #if self.p['product_mode'] else int(agent_string)
        parse['agent']= agent_string_split[0]  #if self.p['product_mode'] else int(agent_string)

        parse['category']= agent_string_split[1] if len(agent_string_split)==3 else None
        parse['product']= int(agent_string_split[2])if len(agent_string_split)==3 else None
        return parse

    def get_truncated_normal(self,mean=0.5, sd=0.2, low=0, upp=1.0):
        rv = truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        return rv

    def order_product_ranks(self):
        current_product_ranking = {}
        for agentstring,rank in self.ranks.items():
            category = self.parse(agentstring)['category']
            product = self.parse(agentstring)['product']
            if category not in current_product_ranking:
                current_product_ranking[category] = []
            current_product_ranking[category].append((product,rank))
        for category, product_tuplist in current_product_ranking.items():
            product_tuplist.sort(key=lambda tup: tup[1],reverse = True)
        self.topNPercent = {}
        self.bottomNPercent = {}
        for category, product_tuplist in current_product_ranking.items():
            topN = int(self.parameters["top_n_percent"] * 0.01 * len(product_tuplist))
            bottomN =  int(self.parameters["bottom_n_percent"] * 0.01 * len(product_tuplist))
            self.topNPercent[category] = set(product_tuplist[:topN])
            self.bottomNPercent[category] = set(product_tuplist[bottomN:])

    def detect_anomaly(self):

        if self.daynum > 1:
            unnorm = []
            for i in range(len(self.schedule.agents)):
                a = self.schedule.agents[i]
                if a.t0 and a.t1:
                    num_categories = len(self.parameters["chance_of_supplying"])
                    evidence = []
                    for j in range(num_categories):
                        stri = str(j)
                        val = 1 if a.t0[stri] else 0
                        evidence.append(val)
                    for j in range(num_categories):
                        stri = str(j)
                        val = 1 if a.t1[stri] else 0
                        evidence.append(val)

                    conformity = self.anomaly_net.log_probability(evidence)
                    unnorm.append(conformity)
                else:
                    unnorm.append(None)
            if len(unnorm)> 1:
                remaining_unnorm = list(filter(None, unnorm))
                min_unnorm = min (remaining_unnorm ) if len(remaining_unnorm) else None
                positive = [i - min_unnorm if i is not None  else None for i in unnorm] if min_unnorm is not None else []
                remaining = list(filter(None, positive))
                max_positive=max(remaining )if len(remaining) else None
                norm = [float(i)/ max_positive if i else None for i in positive] if max_positive else []
                if len(norm)> 0:
                    for i in range(len(self.schedule.agents)):
                        a = self.schedule.agents[i]
                        if norm[i]:
                            a.add_conformity(norm[i])
                            self.conformity_score[a.unique_id] = a.avg_conformity()
        return (self.conformity_score)

    def threshold(self,score_weights, threshold):
        #threhold the socore weights then normalize so sum == 1
        over_threshold = {k:v for k,v in score_weights.items() if v and v > threshold}
        return over_threshold


    def norm(self,score_weights):
        #threhold the socore weights then normalize so sum == 1
        normalized = {}
        if len(score_weights) > 0:
            min_unnorm = min(score_weights.values())
            positive = {k:v - min_unnorm for k,v in score_weights.items()}
            sum_pos = sum(positive.values())
            if sum_pos >0:
                normalized ={k:(v/sum_pos) for k,v in positive.items()}
        return normalized

    def reputation_system_stub(self,score_weights):
        rep_system = {}
        if len(score_weights)> 0:
            current_suppliers = set(
                [supplier for good, supplierlist in self.suppliers.items() for supplier in supplierlist])
            for supplier in current_suppliers:
                supplier_agent = self.agents[self.m[self.orig[supplier]]]
                for category, product_dict in supplier_agent.products.items():
                    for product, product_info_dict in product_dict.items():
                        sum = 0
                        fair_rating = False
                        sum_to_one =self.norm({rater:score_weights[rater] for rater, _ in product_info_dict["ratings"].items(
                            ) if rater in score_weights})
                        for rater, rating in product_info_dict["ratings"].items():
                            if rater in sum_to_one:
                                sum += rating * sum_to_one[rater]
                                fair_rating = True
                        if fair_rating:
                            good_string = "{0}.{1}".format(category, product) if self.parameters['product_mode'] else category
                            reputation_entry = "{0}.{1}".format(supplier, good_string) if self.parameters[
                                'product_mode'] else supplier
                            rep_system[reputation_entry]= int(sum * 100)
                            #if sum > 0 and sum < 1:
                                #print ("product {0} reputation {1}".format(reputation_entry,sum))
        return(rep_system)

    def find_hopfield_goodness(self):
        #to be run n times and averaged to get final score
        num_iter = self.parameters['hopfield_num_iters']
        noise = 0.1
        num_samples = self.parameters['hopfield_num_samples']
        firsttime = True
        for _ in range(num_samples):
            self.hopfield_net.run({self.reputation_mech: [
                np.random.random((1, self.hopfield_size)) * noise
                for t in range(num_iter)
            ]}, num_trials=num_iter)
            all_results = np.squeeze(self.hopfield_net.results)
            final = all_results[-1,:]
            self.hopfield_score = {}
            f=lambda x: 1 if x > 1 else 0
            #print('\n\n')
            for i in range (len(self.agent_labels)):
                self.hopfield_score [self.agent_labels[i]] = final[i] if firsttime else self.hopfield_score [self.agent_labels[i]] + final[i]
                #print('{0}:{1},'.format(self.agent_labels[i],final[i]), end='')
            firsttime = False

        print('\n\n')
        for i in range (len(self.agent_labels)):
            self.hopfield_score [self.agent_labels[i]]/= num_samples
            print('{0}:{1},'.format(self.agent_labels[i],final[i]), end='')

        return self.hopfield_score

    def find_predictiveness(self):

        if self.daynum > self.parameters["days_until_prediction"]+1:
            unnorm = []
            for i in range(len(self.schedule.agents)):
                a = self.schedule.agents[i]
                predictiveness = 0
                if a.t0 and a.t1:
                    num_categories = len(self.parameters["chance_of_supplying"])
                    evidence = []
                    for j in range(num_categories):
                        stri = str(j)
                        val = 1 if a.t0[stri] else 0
                        evidence.append(val)
                    for j in range(num_categories):
                        stri = str(j)
                        val = 1 if a.t1[stri] else 0
                        evidence.append(val)

                    for val in range(num_categories):
                        val_plus1 = val +1
                        case = copy.deepcopy(evidence)
                        case.append(val_plus1)
                        case.append(None)
                        try:
                            description = self.predictiveness_net.predict_proba(case)
                        except ValueError as err:
                            #print( "never seen {0}, err:{1}".format(case,err))
                            continue
                        description_last_entry = len(description) -1
                        result = (json.loads(description[description_last_entry].to_json()))['parameters'][0]
                        strval = str(val_plus1)
                        #prob_of_predicted = result[strval]
                        prob_of_predicted = result[strval] if strval in result else 0.0
                        predictiveness += prob_of_predicted

                    unnorm.append(predictiveness)
                else:
                    unnorm.append(None)
            if len(unnorm)> 1:
                #remaining_unnorm = list(filter(None, unnorm))
                remaining_unnorm = [i for i in unnorm if i is not None]
                min_unnorm = min (remaining_unnorm ) if len(remaining_unnorm) else None
                positive = [i - min_unnorm if i is not None  else None for i in unnorm] if min_unnorm is not None else []
                #remaining = list(filter(None, positive))
                remaining = [i for i in positive if i is not None]
                max_positive=max(remaining )if len(remaining) else None
                norm = [float(i)/ max_positive if i else None for i in positive] if max_positive else []
                if len(norm)> 0:
                    for i in range(len(self.schedule.agents)):
                        if norm[i]:
                            a = self.schedule.agents[i]
                            a.add_predictiveness(norm[i])
                            self.predictiveness_score[a.unique_id] = a.avg_predictiveness()
        return (self.predictiveness_score)


    def get_ranks(self, prev_date):

        if self.parameters['rep_system'] == "aigents" and self.daynum % self.parameters['ranks_update_period'] == 0:
                self.reputation_system.update_ranks(prev_date)

        elif self.parameters['rep_system']=='anomaly' and (self.daynum - 2) % self.parameters["anomaly_net_creation_period"] == 0:
            self.anomaly_net = self.create_anomaly_net()

        elif self.parameters['rep_system']=='hopfield' and (self.daynum - 1) % self.parameters["hopfield_net_creation_period"] == 0:
            self.hopfield_net = self.create_hopfield_net()


        self.orig_ranks = (self.reputation_system_stub(self.threshold(self.detect_anomaly(),self.parameters['conformity_threshold'])
                ) if self.parameters['rep_system'] == "anomaly"
                else (self.reputation_system_stub(self.threshold(self.find_hopfield_goodness(),self.parameters['hopfield_threshold'])
                ) if self.parameters['rep_system'] == "hopfield" else self.reputation_system.get_ranks_dict({'date':prev_date})))
        predictive_ranks = None
        min_days = self.daynum - (2 + self.parameters["days_until_prediction"])
        if (self.parameters['rep_system_booster'] == "predictiveness" and
                min_days > 0 and
                min_days % self.parameters["predictiveness_net_creation_period"] == 0):
            self.predictiveness_net = self.create_predictiveness_net()
            predictive_ranks = self.reputation_system_stub(self.threshold(self.find_predictiveness(),self.parameters['predictiveness_threshold']))

        self.ranks = predictive_ranks if predictive_ranks else self.orig_ranks

        #only put current suppliers and products
        if self.ranks:
            current_suppliers = set([supplier for good, supplierlist in self.suppliers.items() for supplier in supplierlist])

            for agentstring,rank in self.ranks.items():
                agent = self.parse(agentstring)['agent']
                if agent in current_suppliers:
                    category = self.parse(agentstring)['category']
                    product = self.parse(agentstring)['product']
                    supplier_agent = self.agents[self.m[self.orig[agent]]]
                    if product in supplier_agent.products[category]:
                        self.add_rank(agent,rank)

            self.order_product_ranks()
        #generation_increment = (self.model.daynum // self.p['scam_period']) * self.p['num_users']
        #if self.p['scam_inactive_period']:  #there is a campaign in which ids are shifted, so subtract the increment


    def create_hopfield_net_products(self):
        #this is the product version that took too long
        #create 2 sets of labels for raters and for products they rate,
        #sort them into a list
        #and then make a list of list of ratings of ratings in that sorted order
        #raters as cols and products as rows, for a numpy array, mapped to -1 and 1
        product_labels = set()
        rater_labels = set()
        for agent in self.schedule.agents:
            for category, product_dict in agent.products.items():
                for product, product_info_dict in product_dict.items():
                    product_labels.add(product)
                    for rater, rating in product_info_dict["ratings"].items():
                        rater_labels.add(rater)
        self.product_labels = sorted(list(product_labels))
        product_map = {label:num for num,label in enumerate(self.product_labels)}
        self.rater_labels = sorted(list(rater_labels))
        rater_map = {label:num for num,label in enumerate(self.rater_labels)}
        num_rows = len(product_labels)
        num_cols = len(rater_labels)
        ratings = np.zeros((num_rows,num_cols))
        f = lambda x: (-1 if self.parameters['ratings_bayesian_map'][str(x)] - 3 < 0
                       else 0 if self.parameters['ratings_bayesian_map'][str(x)] - 3 == 0
                        else 1)

        for agent in self.schedule.agents:
            for category, product_dict in agent.products.items():
                for product, product_info_dict in product_dict.items():
                    for rater, rating in product_info_dict["ratings"].items():
                        ratings[product_map[product],rater_map[rater]]= f(rating)

        ratings_transpose = ratings.transpose()
        #no need to reinforce the self, it is conservative
        #raters = np.eye(num_cols)
        #products = np.eye(num_rows)
        raters = np.zeros((num_cols,num_cols))
        products = np.zeros((num_rows,num_rows))
        matrix = np.vstack([
            np.hstack([raters,ratings_transpose]),
            np.hstack([ratings,products])
        ])


        function = pnl.Linear
        noise = 0
        integration_rate = .5
        self.hopfield_size = num_rows+num_cols
        self.reputation_mech = pnl.RecurrentTransferMechanism(
            size=self.hopfield_size,
            function=function,
            matrix=matrix,
            integration_rate=integration_rate,
            noise=noise,
            name='reputation rnn'
        )
        reputation_process = pnl.Process(pathway=[self.reputation_mech])
        hopfield_net = pnl.System(processes=[reputation_process])
        return hopfield_net

    def create_hopfield_net(self):
        #this is the supplier judge version that should take shorter
        #create 2 sets of labels for raters and for products they rate,
        #sort them into a list
        #and then make a list of list of ratings of ratings in that sorted order
        #raters as cols and products as rows, for a numpy array, mapped to -1 and 1

        self.agent_labels = [a.unique_id for a in self.schedule.agents]
        agent_map = {label:num for num,label in enumerate(self.agent_labels)}
        num_rows = num_cols = len(self.agent_labels)
        ratings = np.zeros((num_rows,num_cols))
        f = lambda x: (self.parameters['ratings_bayesian_map'][str(x)] - 3 )

        for agent in self.schedule.agents:
            for other_agent in self.schedule.agents:
                if agent.unique_id != other_agent.unique_id:
                    rating = agent.consumer_to_supplier_avg_rating(other_agent.unique_id)
                    if rating is not None:
                        rate = f(rating)
                        ratings[agent_map[agent.unique_id],agent_map[other_agent.unique_id]]= rate
                        ratings[agent_map[other_agent.unique_id],agent_map[agent.unique_id]]= rate


        matrix = np.array(ratings)
        #print("\n\n")
        #print(matrix)

        function = pnl.Linear
        noise = 0
        integration_rate = .5
        self.hopfield_size = num_rows
        self.reputation_mech = pnl.RecurrentTransferMechanism(
            size=self.hopfield_size,
            function=function,
            matrix=matrix,
            integration_rate=integration_rate,
            noise=noise,
            name='reputation rnn'
        )
        reputation_process = pnl.Process(pathway=[self.reputation_mech])
        hopfield_net = pnl.System(processes=[reputation_process])

        return hopfield_net

    def create_anomaly_net(self):
        X = []
        for agent in self.schedule.agents:
#           if any(agent.anomaly_detection_data):
                X.extend(agent.anomaly_detection_data)

        model = BayesianNetwork.from_samples(X,algorithm='chow-liu') if len(X)> 0 else None
        return model

    def create_predictiveness_net(self):
        X = []
        for agent in self.schedule.agents:
            X.extend(agent.predictiveness_data)
        model = BayesianNetwork.from_samples(X, algorithm='chow-liu') if len(X)> 0 else None
        return model

    def step(self):
        present = int(round(self.schedule.time))
       # print('time {0}'.format(present))
        print('.',end='')
        if self.error_log:
            self.error_log.write('time {0}\n'.format(self.schedule.time))
        self.daily_avg_rank = self.calculate_daily_avg_rank()
        self.reset_current_ranks()
        """Advance the model by one step."""
        self.schedule.step()

        self.print_stats_product()
        self.daily_stats_reset()
        self.print_market_volume_report_line()
        #self.market_volume_report.flush()
        if self.error_log:
            self.error_log.flush()
        self.daynum = int(round(self.schedule.time))
        prev_date = self.since + dt.timedelta(days=(self.daynum - 1))
        if not self.parameters["observer_mode"]:
            self.get_ranks(prev_date)
            if self.parameters['product_mode']:
                self.write_current_rank_history_line()
            else:
                self.write_rank_history_line()
            # if self.error_log:
            #     self.error_log.write("ranks: {0}\n".format(str(self.ranks)))

    def go(self):
        while self.schedule.time < self.get_end_tick():
            self.step()
        self.finalize_all_ranks()
        self.write_output_stats_line()
        self.output_stats.flush()
        self.output_stats.close()
        self.market_volume_report.close()
        self.transaction_report.close()
        if self.error_log:
            self.error_log.close()
        if self.rank_history:
            self.rank_history.write(self.rank_history_heading)
            self.rank_history.close()


    def initialize_reputation_system(self,config):

        if config['parameters']['observer_mode']:
            self.reputation_system = None
        elif self.reputation_system is not None:
            self.reputation_system.set_parameters({
                'precision': config['parameters']['reputation_parameters']['precision'],
                'default': config['parameters']['reputation_parameters']['default'],
                'conservatism': config['parameters']['reputation_parameters']['conservatism'],
                'fullnorm': config['parameters']['reputation_parameters']['fullnorm'],
                'weighting': config['parameters']['reputation_parameters']['weighting'],
                'logratings': config['parameters']['reputation_parameters']['logratings'],
                'decayed': config['parameters']['reputation_parameters']['decayed'],
                'liquid': config['parameters']['reputation_parameters']['liquid'],
                'logranks': config['parameters']['reputation_parameters']['logranks'],
                'downrating': config['parameters']['reputation_parameters']['downrating'],
                'update_period': config['parameters']['reputation_parameters']['update_period'],
                'aggregation': config['parameters']['reputation_parameters']['aggregation'],
                'denomination': config['parameters']['reputation_parameters']['denomination'],
                'unrated': config['parameters']['reputation_parameters']['unrated'],
                'ratings': config['parameters']['reputation_parameters']['ratings'],
                'spendings': config['parameters']['reputation_parameters']['spendings'],
                'rating_bias': config['parameters']['reputation_parameters']['rating_bias'],
                'predictiveness': config['parameters']['reputation_parameters']['predictiveness'],
                'parents': config['parameters']['reputation_parameters']['parents']


            })


    def make_reputation_system(self,config):

        now = dt.datetime.now()
        epoch = now.strftime('%s')
        dirname = 'test' + epoch
        rs = None
        if not config['parameters']['observer_mode']:
            rs =  (PythonReputationService(
            ) if not config['parameters']['use_java'] else AigentsAPIReputationService(
                'http://localtest.com:{0}/'.format(config['parameters']['port']),
                'john@doe.org', 'q', 'a', False, dirname, True))

        return rs

class Runner():

    def __init__(self,config):
        self.param_list = []
        self.config = config
        self.output_stats = self.output_stats(config)
        self.output_stats.close()

    def make_reputation_system(self, config):
        now = dt.datetime.now()
        epoch = now.strftime('%s')
        dirname = 'test' + epoch
        rs = None
        if not config['parameters']['observer_mode']:
            rs =  (PythonReputationService(
            ) if not config['parameters']['use_java'] else AigentsAPIReputationService(
                'http://localtest.com:{0}/'.format(config['parameters']['port']),
                'john@doe.org', 'q', 'a', False, dirname, True))

        return rs

    def get_output_stats_header(self):
        heading = []
        heading.append("study")
        for new_val in self.config['output_columns']:
            # old_val = configfile['parameters']
            old_val = self.config
            old_old_val = old_val
            while isinstance(new_val, dict) and len(new_val) == 1:
                # while isinstance(new_val, dict):
                nextKey = next(iter(new_val.items()))[0]
                old_old_val = old_val
                old_val = old_val[nextKey]
                new_val = new_val[nextKey]
            #old_old_val[nextKey] = new_val
            paramlist = new_val
            for param in paramlist:
                length = len(old_old_val[nextKey][param])if isinstance(old_old_val[nextKey][param],list)else 1
                if length > 1:
                    for i in range(length):
                        new_param = "{0}{1}".format(param,i)
                        heading.append(new_param)
                else:
                    heading.append(param)
        for test,_ in self.config['tests']['default'].items():
            heading.append(test)
        heading.append("\n")
        headingstr = "\t".join(map (str, heading))
        return(headingstr)



    def output_stats(self,config):
        path = config['parameters']['output_path'] + 'output_stats.tsv'

        if not os.path.exists(config['parameters']['output_path']):
            os.makedirs(config['parameters']['output_path'])

        with open(path, 'a') as file:
            header = self.get_output_stats_header()
            file.write(header)
        return file



    def createTestCsv(self,config,codelist=None):
        import pandas as pd

        from copy import deepcopy

        outpath = config['parameters']['output_path'] + 'results.csv'
        allcols = ['code', 'folder', 'spendings', 'ratings', 'unrated', 'denom',
                   'logratings', 'fullnorm', 'conserv',
                   'default', 'downrating', 'decayed', 'period',
           #        'precision', 'recall', 'f1', 'satisfaction',
                   'inequity', 'utility',
           #        'pearson_by_good', 'pearsong_by_good', 'pearsonb_by_good',
                   'loss_to_scam', 'profit_from_scam',  'market_volume']
        #
        # allcols = ['code', 'folder', 'spendings', 'ratings', 'unrated', 'denom',
        #            'logratings', 'fullnorm', 'conserv',
        #            'default', 'downrating', 'decayed', 'period',
        #            'loss_to_scam', 'profit_from_scam', 'inequity', 'utility', 'market_volume']

        columns = ['ratings', 'spendings', 'unrated', 'downrating', 'denom',
                   'logratings', 'fullnorm', 'default', 'conserv', 'decayed', 'period','folder', 'code']



        # alist = [
        #     ['0.5', '0.5', 'false', 'false', 'true', 'false', 'true', '0.0', '0.9', '0.5', '1',
        #      'weightDenomNoUnratedConserv9Ratings5Spending5half']
        # ]

        #codelist = ['r_sp182', 'r_sp92', 'r_sp30', 'r_sp10']

        if codelist is None:
            codelist = [code for code, test in config['tests'].items()]

        #codelist = config['tests'].keys()

        testfiles = [
         #   "discrete_rank_tests.tsv",
         #   "correlation_by_good_tests.tsv",
            "scam_loss_tests.tsv",
            "utility_tests.tsv",
         #   "satisfaction_tests.tsv",
            "inequity_tests.tsv",
            "market_volume_tests.tsv",
         #   "price_variance_tests.tsv",
         #   "correlation_tests.tsv",
        #    "continuous_rsmd_tests.tsv",
         #   "continuous_rsmd_by_good_tests.tsv",
            "scam_profit_tests.tsv",
        ]

        dflist = []
        #for row in alist:
        runslist = []
        for code in codelist:
            newconfigpath = config['parameters']['output_path'] + "params_" + code + ".json"

            with open(newconfigpath) as json_file:
                newconfig = json.load(json_file, object_pairs_hook=OrderedDict)
            p = newconfig['parameters']['reputation_parameters']
            paramvals = []
            paramvals.append(str(p['ratings']))
            paramvals.append(str(p['spendings']))
            paramvals.append(str(p['unrated']))
            paramvals.append(str(p['downrating']))
            paramvals.append(str(p['denomination']))
            paramvals.append(str(p['logratings']))
            paramvals.append(str(p['fullnorm']))
            paramvals.append(str(p['default']))
            paramvals.append(str(p['conservatism']))
            paramvals.append(str(p['decayed']))
            paramvals.append(str(p['update_period']))
            paramvals.append(str(p['rating_bias']))
            paramvals.append(str(p['predictiveness']))
            paramvals.append(str(p['parents']))
            paramvals.append(config['parameters']['output_path'][: -1])
            row = paramvals
           # print(row)

            copy = deepcopy(row)
            copy.append(code)
            runslist.append(copy)
            runs = pd.DataFrame(runslist, columns=columns)
        # print(runs)
        for t in testfiles:
            path = row[11] + '/' + t
            try:
                df = pd.read_csv(path, delimiter='\t')
                df['folder'] = row[11]
                #print(path)
                #print(df)
                # if len(df.index)== len(codelist):
                runs = pd.merge(runs, df, on=['folder', 'code'])
                # print(runs)
            except FileNotFoundError as e:
                #print(e)
                pass
            except:
                pass
        #dflist.append(runs)

        #result = pd.concat(dflist)
        #print(result)
        #result = result[allcols]
        #result.to_csv(outpath)
        #result
        result = runs[allcols]
        result.to_csv(outpath)


    def get_param_list(self, combolist, param_str = ""):
        if combolist:
            mycombolist = copy.deepcopy(combolist)
            level,settings = mycombolist.popitem(last = False)
            for name, setting in settings.items():
                my_param_str = param_str + name + "_"
                self.get_param_list(mycombolist,my_param_str)
        else:
            self.param_list.append(param_str[:-1])


    def run_tests(self):
        #test = ContinuousRankByGoodTests()
        #test.go(config)
        #test = ContinuousRankTests()
        #test.go(config)
        # test = DiscreteRankTests()
        # test.go(config)
        test = GoodnessTests()
        test.go(self.config)
        test = MarketVolumeTests()
        test.go(self.config)
        test = TransactionsTests()
        test.go(self.config)
        self.get_param_list(self.config['batch']['parameter_combinations'])
        self.createTestCsv(self.config, set(self.param_list))


    def set_param(self,configfile, setting):
        # setting is OrderedDict, perhaps nested before val is set.  example :  {"prices": {"milk": [2, 0.001, 0, 1000] }}
        #old_val = configfile['parameters']
        old_val = configfile
        new_val = setting
        nextKey = next(iter(new_val.items()))[0]
        old_old_val = old_val
        while isinstance(new_val, dict) and len(new_val)== 1:
        #while isinstance(new_val, dict):
            nextKey = next(iter(new_val.items()))[0]
            old_old_val = old_val
            old_val = old_val[nextKey]
            new_val = new_val[nextKey]
        old_old_val[nextKey] = new_val

    def call(self, combolist, configfile, rs=None,  param_str = ""):
        if combolist:
            mycombolist = copy.deepcopy(combolist)
            level,settings = mycombolist.popitem(last = False)
            for name, settingsList in settings.items():
                myconfigfile = copy.deepcopy(configfile)
                for setting  in settingsList:
                    self.set_param(myconfigfile, setting)
                my_param_str = param_str + name + "_"
                # for sttarting in the middle of a batch run
                #if not (
                        #my_param_str == 'r_20_1_'  or
                        #my_param_str == 'r_20_0.5_' or
                        #my_param_str == 'r_20_0.1_' or
                        #my_param_str == 'r_10_1_' or
                        #my_param_str == 'r_10_0.5_'
                        # my_param_str == 'r_sp182_' or
                        # my_param_str == 'r_sp92_' or
                        #my_param_str == 'r_sp30_'
                        #my_param_str == 'r_norep_' #or
                        #my_param_str == 'r_regular_'
                        #my_param_str == 'r_weighted_' or
                       # my_param_str == 'r_SOM_' or
                        #my_param_str == 'r_TOM_'
                    #my_param_str.endswith("_LE_overlap_") or
                    #my_param_str.endswith("_noLE_overlap_") or
                    #my_param_str.endswith("_LE_NoOverlap_")
                #):
                #if not my_param_str.startswith("r"):

                self.call(mycombolist, myconfigfile, rs, my_param_str)
        else:
            #new_seed = configfile['parameters']['seed'] + 1
            #set_param(configfile, {"seed": new_seed})
            if configfile['parameters']['seed']:
                np.random.seed(seed=configfile['parameters']['seed'])
                random.seed(configfile['parameters']['seed'] )

            self.set_param( configfile,{"parameters": {"param_str": param_str }})
            if configfile['parameters']['macro_view']:
                configfile = Adapters(configfile).translate()
            repsim = ReputationSim(study_path =configfile, rs=rs, opened_config = True)
            #if configfile['parameters']['use_java']:
             #   print ("{0} : {1}  port:{2} ".format(configfile['parameters']['output_path'],param_str,configfile['parameters']['port']))
            print("{0} : {1}".format(configfile['parameters']['output_path'], param_str))

            repsim.go()


def main():

    #print (os.getcwd())
    study_path = sys.argv[1] if len(sys.argv)>1 else 'study.json'
    with open(study_path) as json_file:
        config = json.load(json_file, object_pairs_hook=OrderedDict)
        # if config['parameters']['seed']:
        #     np.random.seed(seed=config['parameters']['seed'])
        #     random.seed(config['parameters']['seed'] )
        #if config['parameters']['macro_view']:
            #config = Adapters(config).translate()
        runner = Runner(config)
        if config['batch']['on']:
            rs = runner.make_reputation_system(config) #if config['parameters']['use_java'] else None
            runner.call(config['batch']['parameter_combinations'],config, rs=rs)
            if config['parameters']["run_automatic_tests"]:
                runner.run_tests()
            #runner.output_stats.close()
        else:
            repsim = ReputationSim(sys.argv[1]) if len(sys.argv) > 1 else ReputationSim()
            repsim.go()

if __name__ == '__main__':
    main()
