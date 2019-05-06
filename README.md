# reputation-simulation
Market simulation to test the Reputation System 


## Development 

* Clone repository and enter in folder

```sh
$ git clone https://github.com/singnet/reputation-simulation && cd reputation-simulation
```


* Install dependencies from inside reputation-simulation directory

```sh
$ bash scripts/install
```

If you get an error about `mesa/time` package, launch to be sure to install the right `mesa` fork.

```sh
$ pip install -e git+https://github.com/deborahduong/mesa.git#egg=mesa
```

## Usage

```sh
$ python3 snsim/reputation_simulation/ReputationSim.py snsim/rn.json
```

