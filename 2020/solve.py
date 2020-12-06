import fire
import itertools
import functools
import re
from collections import Counter
from rich import print

class AdventOfCode:

    def __init__(self):
        pass
    
    # day 01

    def find_two_entries_that_sum_to(self, value: int, puzzle_input: list) -> tuple:
        candidate_entries = filter(lambda x: x <= value, puzzle_input)
        candidate_entries = dict((entry, value - entry) for entry in sorted(candidate_entries))
        for entry, complement in candidate_entries.items():
            if complement in candidate_entries:
                return entry, complement

    def find_three_entries_that_sum_to(self, value: int, puzzle_input: list) -> tuple:
        candidate_entries = sorted(filter(lambda x: x <= value, puzzle_input))
        complements = dict((entry, value - entry) for entry in candidate_entries)
        possible_sum = dict(((x, y), x + y) for x, y in itertools.product(candidate_entries, candidate_entries) if x != y)
        for pair, sum_ in possible_sum.items():
            if sum_ in complements.values():
                return (*pair, value - sum_)

    # day 02

    def check_if_valid_password(self, policy_password: tuple, new_interpretation: bool) -> bool:

        policy, password = policy_password[:3], policy_password[-1]

        if new_interpretation:
            i1, i2, letter = policy
            occurrences = Counter(password[i1-1] + password[i2-1]).get(letter, 0)
            return occurrences == 1
        else:
            min_, max_, letter = policy
            occurrences = Counter(password).get(letter, 0)
            reach_min = min_ <= occurrences
            below_max = max_ >= occurrences
            return reach_min and below_max

    def count_valid_password(self, puzzle_input: list, new_interpretation: bool = False) -> int:
        return sum(
                list(
                    map(
                        lambda x: self.check_if_valid_password(x, new_interpretation=new_interpretation),
                        puzzle_input
                    )
                )
            )

    # day 03

    def move(self, position: tuple, width: int, slope: tuple) -> tuple:
        y, x = position
        down, right = slope
        return (y + down, (x + right)%width)

    def compute_toboggan_path(self, puzzle_input: list, slope: tuple) -> list:
        width = len(puzzle_input[0])
        height = len(puzzle_input)

        path = [(0,0)]

        while path[-1][0] < height - 1:
            path.append(
                self.move(path[-1], width, slope=slope)
            )
        return path

    def trace_toboggan_path(self, puzzle_input: list, slope: tuple) -> list:
        trace = puzzle_input.copy()
        trace = list(list(x) for x in trace)

        for y, x in self.compute_toboggan_path(puzzle_input, slope=slope):
            if puzzle_input[y][x] == '.':
                trace[y][x] = 'O'
            elif puzzle_input[y][x] == '#':
                trace[y][x] = 'X'

        trace = list(''.join(x) for x in trace)
        return trace

    def count_trees_encountered(self, puzzle_input: list, slope: tuple = (1, 3)) -> int:
        return sum(
            puzzle_input[y][x] == '#'
            for y, x in self.compute_toboggan_path(puzzle_input, slope=slope)
        )

    def evaluate_toboggan_slopes(self, puzzle_input: list, slopes: list) -> int:
        encounters = []
        for slope in slopes:
            encounters.append(self.count_trees_encountered(puzzle_input, slope=slope))
        return functools.reduce(lambda x, y: x*y, encounters)

    # day 04

    def fields_missing_from_passport(self, passport: dict, required_fields: list) -> set:
        return set(required_fields) - set(passport.keys())

    def check_if_valid_passport(self, passport: dict, required_fields: list, optional_fields: list) -> bool:
        missing_fields = self.fields_missing_from_passport(passport, required_fields)
        return missing_fields.issubset(set(optional_fields))

    def count_valid_passports(self, puzzle_input: list) -> int:
        return sum(
            self.check_if_valid_passport(
                passport, 
                ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'], 
                ['cid']
                )
            for passport in puzzle_input
        )

    def get_valid_passports(self, puzzle_input: list) -> list:
        return list(
            filter(
                lambda x: self.check_if_valid_passport(
                    x, 
                    ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'], 
                    ['cid']
                    ),
                puzzle_input
                )                
            )

    def fields_validation(self) -> dict:
        return {
            'byr': lambda x: len(x) == 4 and int(x) >= 1920 and int(x) <= 2002 if x.isdigit() else False,
            'iyr': lambda x: len(x) == 4 and int(x) >= 2010 and int(x) <= 2020 if x.isdigit() else False,
            'eyr': lambda x: len(x) == 4 and int(x) >= 2020 and int(x) <= 2030 if x.isdigit() else False,
            'hgt': (
                lambda x: int(x.replace('cm','')) >= 150 and int(x.replace('cm','')) <= 193 
                if 'cm' == x[-2:] and x.replace('cm','').isdigit()
                else int(x.replace('in','')) >= 59 and int(x.replace('in','')) <= 76 
                if 'in' == x[-2:] and x.replace('in','').isdigit() 
                else False
                ), 
            'hcl': lambda x: len(x) == 7 and x[0] == '#' and all(y in list('abcdef0123456789') for y in list(x[1:])), 
            'ecl': lambda x: x in ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth'], 
            'pid': lambda x: len(x) == 9 and x.isdigit(),
            'cid': lambda x: True
        }
        
    def validate_passport_data(self, passport: dict) -> bool:
        return all(self.fields_validation()[field](passport[field]) for field in passport)

    def count_passports_with_valid_data(self, puzzle_input: list) -> int:
        return sum(
            self.validate_passport_data(passport)
            for passport in puzzle_input
        )

    # day 05

    def decode_seat(self, chunk, keys):
        lb, ub = 0, 2 ** len(chunk) - 1
        for char in chunk:
            middle = int((ub - lb + 1) / 2)
            if char == keys[0]:
                lb, ub = lb, lb + middle - 1
            elif char == keys[1]:
                lb, ub = lb + middle, ub
        if lb == ub:
            return lb        

    def compute_seat_id(self, boarding_pass):
        row = self.decode_seat(boarding_pass[:7], ['F', 'B'])
        column = self.decode_seat(boarding_pass[7:], ['L', 'R'])
        return row * 8 + column

    def compute_highest_seat_id(self, puzzle_input):
        return max(self.compute_seat_id(boarding_pass) for boarding_pass in puzzle_input)

    def find_seat(self, puzzle_input):
        seats = list(self.compute_seat_id(bp) for bp in puzzle_input)
        return set(range(min(seats), max(seats) + 1)) - set(seats)

    # day 06

    def get_group_questions(self, group, rule):
        if rule == 'anyone':
            return len(functools.reduce(
                lambda x, y: set(x) | set(y),
                group
            ))
        elif rule == 'everyone':
            return len(functools.reduce(
                lambda x, y: set(x) & set(y),
                group
            ))

    def compute_questions_count(self, puzzle_input, rule='anyone'):
        return sum(
            self.get_group_questions(group, rule=rule)
            for group in puzzle_input
        )


def load_puzzle_input(day: str) -> str:
    with open(f'./puzzle_input/{day}.txt', 'r') as f:
        puzzle_input = f.read()
    return puzzle_input

def preprocess(puzzle_input: str, day: str) -> list:

    def _standard_preprocess(puzzle_input):
        return filter(lambda x: x != '', puzzle_input.split('\n'))

    if day == '01':
        puzzle_input = _standard_preprocess(puzzle_input)
        return list(
            map(
                lambda x: int(x),
                puzzle_input
                )
            )
    if day == '02':
        puzzle_input = _standard_preprocess(puzzle_input)
        return list(
            map(
                lambda x: (
                    int(x[0].split('-')[0]),
                    int(x[0].split('-')[1]),
                    x[1].replace(':',''),
                    x[2]
                    ),
                map(
                    lambda x: x.split(' '),
                    puzzle_input
                )
            )
        )
    if day == '03':
        return list(_standard_preprocess(puzzle_input))
    if day == '04':
        puzzle_input = map(lambda x: x.split('\n'), puzzle_input.split('\n\n'))
        puzzle_input = map(lambda x: list(y.split(' ') for y in x), puzzle_input)
        puzzle_input = map(lambda z: functools.reduce(lambda x, y: x + y, z), puzzle_input)
        puzzle_input = map(lambda x: dict(tuple(y.split(':')) for y in x if y != ''), puzzle_input)
        return list(puzzle_input)
    if day == '05':
        return list(_standard_preprocess(puzzle_input))
    if day == '06':
        puzzle_input = puzzle_input.split('\n')
        groups = []
        group = []
        for person in puzzle_input:
            if person != '':
                group.append(person)
            else:
                groups.append(group)
                group = []                
        return groups

def solve(day: str):

    puzzle_input = preprocess(load_puzzle_input(day), day)
    print(f"\n- Day {int(day)} -")
    aoc = AdventOfCode()

    if day in '01':
        n, m = aoc.find_two_entries_that_sum_to(2020, puzzle_input)
        p, q, r = aoc.find_three_entries_that_sum_to(2020, puzzle_input)
        print(f"The two entries are {n} and {m} and their product is {n * m}.")
        print(f"The three entries are {p}, {q} and {r} and their product is {p * q * r}.")
    if day == '02':
        n = aoc.count_valid_password(puzzle_input)
        m = aoc.count_valid_password(puzzle_input, new_interpretation=True)
        print(f"There are {n} valid passwords.")
        print(f"With the new interpretation, there are {m} valid passwords.")
    if day == '03':
        n = aoc.count_trees_encountered(puzzle_input)
        m = aoc.evaluate_toboggan_slopes(puzzle_input, [(1,1),(1,3),(1,5),(1,7),(2,1)])
        print(f"The trees encountered would be {n}.")
        print(f"The product of all encounters is {m}.")
    if day == '04':
        n = aoc.count_valid_passports(puzzle_input)
        m = aoc.count_passports_with_valid_data(aoc.get_valid_passports(puzzle_input))
        print(f"There are {n} passports which contain required fields.")
        print(f"There are {m} passports which contain required fields with valid data.")
    if day == '05':
        n = aoc.compute_highest_seat_id(puzzle_input)
        m = aoc.find_seat(puzzle_input).pop()
        print(f"The highest seat_id is {n}.")
        print(f"The seat id is {m}.")
    if day == '06':
        n = aoc.compute_questions_count(puzzle_input)
        m = aoc.compute_questions_count(puzzle_input, rule='everyone')
        print(f"The sum of questions to which anyone answered yes is {n}.")
        print(f"The sum of questions to which everyone answered yes is {m}.")

def main(day: str):
    print(f"\n*** Advent of Code 2020 ***")
    if day == 'all':
        days = [f"{str(day).zfill(2)}" for day in range(1, 26)]
        for day in days:
            try:
                solve(day)
            except:
                pass
    else:
        solve(day)
    print('\n')

if __name__ == '__main__':
    fire.Fire(main)
