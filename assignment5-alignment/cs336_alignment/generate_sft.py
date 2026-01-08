import argparse
import json

from drgrpo_grader import extract_answer, r1_zero_reward_fn

def get_response(solution: str) -> str:
    answer = extract_answer(solution)
    if answer is None:
        return None
    return f" {solution} </think> <answer> {answer} </answer>"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data")
    parser.add_argument("--output_data")
    parser.add_argument("--prompt_template")

    args = parser.parse_args()
    with open(args.prompt_template, "r") as f:
        prompt_template = f.read()
    
    with open(args.input_data, "r") as f_in, open(args.output_data, "w") as f_out:
        for line in f_in:
            sample = json.loads(line)
            prompt = prompt_template.format(question=sample["problem"])
            response = get_response(sample["solution"])
            if response is None:
                continue
            rewards = r1_zero_reward_fn(response, sample["solution"])
            if rewards["reward"] == 1.0:
                f_out.write(json.dumps({
                    "prompt": prompt,
                    "response": response,
                }) + "\n")

