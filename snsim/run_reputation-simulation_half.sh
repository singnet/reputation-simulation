cd /home/reputation/snsim/reputation
source activate newestrepenv
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py s2NoRepHalf_debug100.json
cp s2NoRepHalf_debug100.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
python3 ~/reputation-simulation/snsim/reputation_simulation/ReputationSim.py s2RepHalf_debug100.json
cp s2RepHalf_debug100.json test.json
python -m unittest discover -s /home/reputation/snsim/reputation/test -t /home/reputation/snsim/reputation/test -p *Tests*
