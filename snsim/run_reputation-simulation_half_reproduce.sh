#start from reputation-simulation, in correct environment
#before running do a chmod +x run_reputation-simulation_half_reproduce.sh
#to remove any windows /r, run  sed -i 's/\r//g' run_reputation-simulation_half_reproduce.sh
cd snsim
python3 reputation_simulation/ReputationSim.py s2NoRepHalf_debug100.json
cp s2NoRepHalf_debug100.json test.json
python -m unittest discover -s test -t test -p *Tests*
python3 reputation_simulation/ReputationSim.py s2RepHalf_debug100.json
cp s2RepHalf_debug100.json test.json
python -m unittest discover -s test -t test -p *Tests*
python3 reputation_simulation/ReputationSim.py weightDenomUnratedConserv9Ratings5Spending5half_debug100.json
cp weightDenomUnratedConserv9Ratings5Spending5half_debug100.json test.json
python -m unittest discover -s test -t test -p *Tests*
