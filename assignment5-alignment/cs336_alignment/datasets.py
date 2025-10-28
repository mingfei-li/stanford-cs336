from datasets import load_dataset


class GSM8KTestDataset:
    def __init__(self):
        with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
            template = f.read()
        ds = load_dataset("openai/gsm8k", "main")
        self.prompts = [
            template.replace("{question}", question)
            for question in ds["test"]["question"]
        ]
        self.ground_truths = [
            answer.split("####")[-1].strip() for answer in ds["test"]["answer"]
        ]
