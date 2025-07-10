import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
from nanovllm.engine.model_runner import lp
# from nanovllm.layers.sampler import lp_sampler

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    path = os.path.expanduser("/root/LLaMA-Factory/saves/qwen3-32b/full/too_call_sft_1000")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=2, enable_tools=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    prompts = [
        "What is the weather in Beijing and Tokyo?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather in a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "The city to get the weather for",
                                }
                            },
                        },
                    },
                }
            ],
        )
        for prompt in prompts
    ]

    iters = 10
    total_time = 0
    total_tokens = 0
    for i in range(iters):
        begin = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end = time.time()
        total_time += end - begin
        total_tokens += len(outputs[0]["text"])
        print(f"Time: {end - begin} seconds")
    print(f"Average time: {total_time / iters} seconds")
    print(f"Average tokens: {total_tokens / iters}")

    # for prompt, output in zip(prompts, outputs):
    #     print("\n")
    #     print(f"Prompt: {prompt!r}")
    #     print(f"Completion: {output['text']!r}")
        
    lp.print_stats()


if __name__ == "__main__":
    main()
