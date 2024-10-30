import re
import argparse


def remove_version_numbers(input_file, output_file):
    with open(input_file, "r", encoding="utf-16-le") as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove version numbers using regex
        cleaned_line = re.sub(r"[=<>!~]=.*", "", line).strip()
        cleaned_lines.append(cleaned_line)

    with open(output_file, "w", encoding="utf-16-le") as file:
        file.write("\n".join(cleaned_lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove version numbers from requirements.txt"
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input requirements.txt file"
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output requirements.txt file"
    )

    args = parser.parse_args()

    remove_version_numbers(args.input_file, args.output_file)
