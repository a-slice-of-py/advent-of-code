import itertools

def load_input(day):
    with open(f'./input/{day}.txt', 'r') as f:
        input = f.read()
    return input

def preprocess(input):
    return map(
            lambda x: int(x),
            filter(lambda x: x != '', input.split('\n'))
            )

def find_two_entries_that_sum_to(value):
    candidate_entries = filter(lambda x: x <= value, preprocess(load_input('01')))
    candidate_entries = dict((entry, value - entry) for entry in sorted(candidate_entries))
    for entry, complement in candidate_entries.items():
        if complement in candidate_entries:
            return entry, complement

def find_three_entries_that_sum_to(value):
    candidate_entries = sorted(filter(lambda x: x <= value, preprocess(load_input('01'))))
    complements = dict((entry, value - entry) for entry in candidate_entries)
    possible_sum = dict(((x, y), x + y) for x, y in itertools.product(candidate_entries, candidate_entries) if x != y)
    for pair, sum_ in possible_sum.items():
        if sum_ in complements.values():
            return *pair, value - sum_

def main():
    n, m = find_two_entries_that_sum_to(2020)
    p, q, r = find_three_entries_that_sum_to(2020)
    print(f"The two entries are {n} and {m} and their product is {n * m}!")
    print(f"The three entries are {p}, {q} and {r} and their product is {p * q * r}!")

if __name__ == '__main__':
    main()
