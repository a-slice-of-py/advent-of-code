# AoC 2019 - part 1

import numpy as np
from rich import print

def mass_to_fuel(mass):
    return int(np.floor(mass/3.0) - 2)

def compute_total_fuel():
    with open('./input.txt', 'r') as f:
        input = f.read()
    return sum(
    	map(
    		lambda x: mass_to_fuel(int(x)),
    		filter(lambda x: x != '', input.split('\n'))
    		)
    	)

def main():
    print(compute_total_fuel())

if __name__ == '__main__':
    main()
