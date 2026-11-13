"""F0 direct: one tool-free call. The system prompt and output cap belong to each benchmark's protocol."""


def run(port, system, user, *, max_output_tokens, timeout_seconds):
    return port.complete([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}], [],
                         max_output_tokens=max_output_tokens, timeout_seconds=timeout_seconds)


SPEC = dict(messages=['system', 'user'], tools=[])
