import json
import os
import pickle
import sys
import re
import random
import numpy as np
from collections import OrderedDict
import copy
import datetime as dt
import time
import operator
from scipy.stats import truncnorm
from pomegranate import *

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

            self.agents = {}
            for good, num_suppliers in nsuppliers.items():
                for _ in range(num_suppliers):
                    agent_str = str(agent_count) + "-0"
                    a = globals()['ReputationAgent'](agent_str, self, criminal=True, supply_list=[good])
                    self.schedule.add(a)
                    self.agents[agent_count]=a
                    self.criminal_suppliers[good].add(agent_str)
                    self.original_suppliers.add(agent_str)
                    self.orig[agent_str]= agent_str
                    self.m[agent_str]= agent_count
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

            for good, num_suppliers in nsuppliers.items():
                for _ in range(num_suppliers):
                    agent_str = str(agent_count) + "-0"
                    a = globals()['ReputationAgent'](agent_str, self, criminal=False, supply_list=[good])
                    self.schedule.add(a)
                    self.agents[agent_count]=a
                    self.orig[agent_str]= agent_str
                    self.m[agent_str]= agent_count
                    agent_count += 1

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
            while len(cpts) and isa_dag:
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
        # num_changes
        #
        self.bsl_num = 0
        self.bsl_denom = 0
        self.sgp_denom = 0
        self.sgl_num_num = 0
        self.sgl_denom_num = 0
        self.sgl_num_denom = 0
        self.sgl_denom_denom = 0
        self.num_changes = 0
        self.sum_good_consumer_ratings = 0
        self.num_good_consumer_rated_purchases = 0

    def add_good_consumer_rating(self, rating):
        self.sum_good_consumer_ratings += rating
        self.num_good_consumer_rated_purchases += 1


    def add_organic_buy(self,price,quality):
        # bsl_num = Σorganicbuys(Price * (1 - OQ))
        # bsl_denom = Σorganicbuys(Price)
        # sgl_denom_num = (Σorganicbuys(Price * OQ))
        # sgl_denom_denom =  Σorganicbuys(OQ)
        self.bsl_num += price * (1.0-quality)
        self.bsl_denom += price
        self.sgl_denom_num += price * quality
        self.sgl_denom_denom += quality

    def add_legit_transaction(self, supplier, price):
        real_supplier = self.orig[supplier]
        if not real_supplier in self.individual_agent_market_volume:
            self.individual_agent_market_volume[real_supplier]=0
        self.individual_agent_market_volume[real_supplier] +=price


    def add_sponsored_buy(self,price, quality, commission):
        # sgp_denom = Σsponsoredbuys(Price * (1 + CR))
        # sgl_num_num = (Σsponsoredbuys(Price * (1 + CR) * OQ))
        # sgl_num_denom = Σsponsoredbuys(OQ)
        self.sgp_denom += commission
        self.sgl_num_num += commission * quality
        self.sgl_num_denom += quality

    def add_identity_change(self):
        self.num_changes += 1

    def bsl(self):
        bsl = self.bsl_num / self.bsl_denom if self.bsl_denom != 0 else -1
        return bsl

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
        asp = (len(criminals) * self.get_end_tick()) / self.num_changes if self.num_changes != 0 else -1
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
        bsl = self.bsl()
        sgp = self.sgp()
        sgl = self.sgl()
        asp = self.asp()
        omut = self.omut()
        market_volume = self.market_volume()
        maxproduct = self.maxproduct()
        lts = self.loss_to_scam()
        pfs = self.profit_from_scam()
        utility = self.utility()
        inequity = self.inequity()

        if self.daynum % 30 == 0:
            print("""\n time:{11}, bsl:{0:.4f}, sgp:{1:.4f}, sgl:{2:.4f}, asp:{3:.4f}, omut:{4:.4f}, market volume:{5:.4f}, maxproduct:{6:.4f}, lts:{7:.4f}, pfs:{8:.4f}, utility:{9:.4}, inequity:{10:.4f}\n""".format(
            bsl,sgp,sgl,asp,omut,market_volume,maxproduct,lts,pfs,utility,inequity,self.daynum))
        if self.error_log:
            self.error_log.write("""\n time:{11}, bsl:{0:.4f}, sgp:{1:.4f}, sgl:{2:.4f}, asp:{3:.4f}, omut:{4:.4f}, market volume:{5:.4f}, maxproduct:{6:.4f}, lts:{7:.4f}, pfs:{8:.4f}, utility:{9:.4}, inequity:{10:.4f}\n""".format(
            bsl,sgp,sgl,asp,omut,market_volume,maxproduct,lts,pfs,utility,inequity,self.daynum))


    def write_output_stats_line(self):
        line = []
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
                        line.append(old_old_val[nextKey][param][i])
                else:
                    line.append(old_old_val[nextKey][param])
        for test,_ in self.config['tests']['default'].items():
            v=""
            if test == "OMUT":
                v = self.omut()
            elif test == "maxproduct":
                v = self.maxproduct()
            elif test == "bsl":
                v = self.bsl()
            elif test == "sgp":
                v = self.sgp()
            elif test == "sgl":
                v = self.sgl()
            elif test == "asp":
                v = self.asp()
            elif test == "loss_to_scam":
                v = self.loss_to_scam()
            elif test == "profit_from_scam":
                v = self.profit_from_scam()
            elif test == "market_volume":
                v = self.market_volume()
            elif test == "inequity":
                v = self.inequity()
            elif test == "utility":
                v = self.utility()
            line.append(v)
        line.append("\n")
        linestr = "\t".join(map (str, line))
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

    def get_ranks(self, prev_date):
        self.ranks = self.reputation_system.get_ranks_dict({'date':prev_date})

        #only put current suppliers and products
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
        self.print_market_volume_report_line()
        #self.market_volume_report.flush()
        if self.error_log:
            self.error_log.flush()
        self.daynum = int(round(self.schedule.time))
        prev_date = self.since + dt.timedelta(days=(self.daynum - 1))
        if self.reputation_system:
            if self.daynum % self.parameters['ranks_update_period'] == 0:
                self.reputation_system.update_ranks(prev_date)
            #if present > 60:
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
