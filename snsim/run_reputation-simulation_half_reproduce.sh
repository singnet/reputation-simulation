#start from reputation-simulation, in correct environment
#before running do a chmod +x run_reputation-simulation_half_reproduce.sh
#to remove any windows /r, run  sed -i 's/\r//g' run_reputation-simulation_half_reproduce.sh
cd snsim
python3 reputation_simulation/ReputationSim.py noIndividuality1_Weighted.json
python3 reputation_simulation/ReputationSim.py noIndividuality1_TOM.json
