"""Single source of built-in, non-permission runtime defaults."""


DEFAULT_PROFILE = "research-evidence-bound"

DEFAULT_ENGLISH_INSTRUCTIONS = (
    "Respond in English only in explanations, public plans, tool query narration, and final answers, "
    "regardless of the input language. Preserve scientific symbols, proper names, and source IDs "
    "exactly as needed. This language instruction does not change tool permissions, evidence "
    "requirements, or other session controls."
)


RUNTIME_DEFAULTS = {
    "request_output_tokens": 4096,
    "request_timeout_seconds": 120.0,
    "max_steps": 24,
    "max_input_tokens": 120000,
    "max_output_tokens": 24000,
    "wall_time_seconds": 600.0,
}
