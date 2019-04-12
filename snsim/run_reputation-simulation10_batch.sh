cd /home/reputation/snsim/reputation
source activate newestrepenv
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py noIndividuality10NoRep.json 
cp noIndividuality10NoRep.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py noIndividuality10_Regular.json 
cp noIndividuality10_Regular.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py noIndividuality10_Weighted.json 
cp noIndividuality10_Weighted.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py noIndividuality10_TOM.json 
cp noIndividuality10_TOM.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py noIndividuality10_SOM.json 
cp noIndividuality10_SOM.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
