import csv
import re
import argparse
import os

def extract_data(text):
    samples = text.split('--------------------------------------------------')
    data = []
    for sample in samples:
        if sample.strip():
            question_match = re.search(r'Question:\n(.*?)\n', sample, re.DOTALL)
            generated_answer_match = re.search(r'Generated answer:\n(.*?)\n', sample, re.DOTALL)
            ground_truth_match = re.search(r'Ground truth answer:\n(.*?)\n', sample, re.DOTALL)
            if question_match and generated_answer_match and ground_truth_match:
                question = question_match.group(1).strip()
                generated_answer = generated_answer_match.group(1).strip()
                ground_truth = ground_truth_match.group(1).strip()
                data.append([question, generated_answer, ground_truth])
    return data

def write_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Question', 'Generated Answer', 'Ground Truth Answer'])
        writer.writerows(data)

def read_input_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def process_file(input_file, output_file):
    try:
        input_text = read_input_file(input_file)
        extracted_data = extract_data(input_text)
        write_to_csv(extracted_data, output_file)
        print(f"Data has been extracted from {input_file} and written to {output_file}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.endswith('.txt'):
                input_file = os.path.join(args.input, filename)
                output_file = os.path.join(args.input, f"{os.path.splitext(filename)[0]}.csv")
                if os.path.isdir(args.output):
                    output_file = os.path.join(args.output, f"{os.path.splitext(filename)[0]}.csv")
                process_file(input_file, output_file)
    else:
        output_file = args.output if args.output else f"{os.path.splitext(args.input)[0]}.csv"
        process_file(args.input, output_file)

if __name__ == "__main__":
    main()
