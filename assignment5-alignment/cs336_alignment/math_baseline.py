from vllm import LLM, SamplingParams
import pandas as pd
import pyarrow.parquet as pq

with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
    template = f.read()
df = pd.read_parquet("data/gsm8k_main_test.parquet", engine="pyarrow")
prompts = [template.replace("{question}", question) for question in df["question"]][:5]

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)

llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")