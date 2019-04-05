#begin in reputation-simulation
cd snsim
python3 reputation_simulation/ReputationSim.py s2NoRepHalf_debug100.json
cp s2NoRepHalf_debug100.json test.json
python -m unittest discover -s test -t test -p *Tests*
python3 reputation_simulation/ReputationSim.py s2RepHalf_debug100.json
cp s2RepHalf_debug100.json test.json
python -m unittest discover -s test -t test -p *Tests*
