import streamlit as st

days = {
    1: "Trebuchet?!",
}


def load_input(day: int) -> list:
    try:
        with open(f"./2023/inputs/{str(day).zfill(2)}.txt", "r") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        pass


def solve_day_1(puzzle_input: list, part: int) -> int:
    if part == 1:
        with st.echo():
            solution = sum(
                int(
                    "".join(
                        [c for c in line if c.isdigit()][idx]
                        for idx in (0, -1)
                    )
                )
                for line in puzzle_input
            )
    else:
        with st.echo():
            digits = {
                "one": "1",
                "two": "2",
                "three": "3",
                "four": "4",
                "five": "5",
                "six": "6",
                "seven": "7",
                "eight": "8",
                "nine": "9"
            }
            def replacer(line: str) -> str:
                for digit in digits:
                    line = line.replace(digit, f"{digit}{digits[digit]}{digit}")
                return line
            solution = solve_day_1(list(map(replacer, puzzle_input)), part=1)
    return solution


solvers = {
    1: solve_day_1
}


def solve_puzzle(day: int, puzzle_input: list, part: int) -> None:
    return solvers.get(day, lambda *args, **kwargs: None)(puzzle_input, part)


def display_puzzle(day: int, title: str) -> None:
    with st.expander(f"**Day {day}: {title}**"):
        puzzle_input = load_input(day)
        if st.toggle("Show input", key=day):
            st.caption(puzzle_input[:10])
        left, right = st.columns(2)
        with left:
            st.markdown("#### Part One")
            st.markdown(f"#### `{solve_puzzle(day, puzzle_input, part=1)}`")
        with right:
            st.markdown("#### Part Two")
            st.markdown(f"#### `{solve_puzzle(day, puzzle_input, part=2)}`")


def main() -> None:
    st.set_page_config(layout="wide")
    _, center, _ = st.columns(3)
    with center:
        st.title("🎄 Advent of Code 2023")
    for day, title in days.items():
        display_puzzle(day, title)

if __name__ == "__main__":
    main()