import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    path = os.path.expanduser("/root/models/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=2, enable_tools=True)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
    prompts = [
        "What is the weather in Beijing and Tokyo?",
        # "What is the weather in Beijing?",
        # "list all prime numbers within 100",
        # """What is the weather in Beijing and Tokyo?
        # <tool_call>
        # {"name": "get_weather", "arguments": {"city": "Beijing"}}
        # </tool_call>
        # <tool_response>
        # {"city": "Beijing", "temperature": 10, "condition": "晴", "humidity": 58, "wind_speed": 0, "timestamp": "2025-06-30 03:04:54"}
        # </tool_response>
        # Next, tool call has been finished, answer the question according to the tool response
        # <tool_call>
        # {"name": "get_weather", "arguments": {"city": "Tokyo"}}
        # </tool_call>
        # <tool_response>
        # {"city": "Tokyo", "temperature": 26, "condition": "阴", "humidity": 63, "wind_speed": 2, "timestamp": "2025-06-30 03:04:55"}
        # </tool_response>
        # Next, tool call has been finished, answer the question according to the tool response""",
        """
        <|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": "Get the weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city to get the weather for"}}}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat is the weather in Beijing and Tokyo?<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user is asking for the weather in both Beijing and Tokyo. Let me check the tools available. There\'s a get_weather function that takes a city parameter. Since the user mentioned two cities, I need to call this function twice, once for each city. First, I\'ll handle Beijing. Then I\'ll do the same for Tokyo. I should make sure each tool call is properly formatted in JSON within the tool_call tags. Let me structure each call separately so the responses can be retrieved correctly.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_response>\n{\n  "city": "Beijing",\n  "temperature": -1,\n  "condition": "小雨",\n  "humidity": 78,\n  "wind_speed": 0,\n  "timestamp": "2025-06-30 04:31:50"\n}\n</tool_response>\nNext, according to the tool response: \n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n<think>
        """,
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

    prompts = [
        """<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": "Get the weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city to get the weather for"}}}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat is the weather in Beijing and Tokyo?<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, the user is asking for the weather in both Beijing and Tokyo. Let me check the tools available. There\'s a get_weather function that takes a city parameter. Since the user mentioned two cities, I need to call this function twice, once for each city. First, I\'ll handle Beijing. Then I\'ll do the same for Tokyo. I should make sure each tool call is properly formatted in JSON within the tool_call tags. Let me structure each call separately so the responses can be retrieved correctly.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n</tool_call>\n<tool_response>\n{\n  "city": "Beijing",\n  "temperature": -1,\n  "condition": "小雨",\n  "humidity": 78,\n  "wind_speed": 0,\n  "timestamp": "2025-06-30 04:31:50"\n}\n</tool_response>\nNext, according to the tool response:\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>\n<tool_response>\n{\n  "city": "Tokyo",\n  "temperature": 7,\n  "condition": "雪",\n  "humidity": 35,\n  "wind_speed": 9,\n  "timestamp": "2025-06-30 06:53:05"\n}\n</tool_response>\nNext, according to the tool response:"""
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
