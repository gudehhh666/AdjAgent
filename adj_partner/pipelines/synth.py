"""F5 synth execution: an ``adjuvant``-profile session without network, with sandboxed code execution.

``decision`` answers closed-evidence decision items from the prompt alone; ``computation`` computes over files
under one granted input directory.
"""
from .sessions import drive

OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS = 6000, 180
TRACKS = {
    'decision': {'profile': 'adjuvant', 'allow_network': False,
                 'allowed_tools': ['update_plan', 'code_interpreter', 'artifact_read', 'job_status', 'research_state'],
                 'budget': {'max_steps': 8, 'max_input_tokens': 120000, 'max_output_tokens': OUTPUT_TOKENS,
                            'wall_time_seconds': 420}},
    'computation': {'profile': 'adjuvant', 'allow_network': False,
                    'allowed_tools': ['update_plan', 'code_interpreter', 'artifact_read', 'job_status', 'file_inspect'],
                    'budget': {'max_steps': 12, 'max_input_tokens': 300000, 'max_output_tokens': OUTPUT_TOKENS,
                               'wall_time_seconds': 600}},
}


def session_config(track, input_root=None):
    """Keyword arguments for ``AgentService.create_session``."""
    if track not in TRACKS:
        raise ValueError('unknown synth track ' + str(track))
    if (track == 'computation') != (input_root is not None):
        raise ValueError('computation needs exactly one input_root; decision takes none')
    cfg = {key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
           for key, value in TRACKS[track].items()}
    cfg['allowed_roots'] = [str(input_root)] if input_root is not None else []
    return cfg


def service_options():
    return dict(request_output_tokens=OUTPUT_TOKENS, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS)


def run(service, track, prompt, *, input_root=None):
    session = service.create_session(**session_config(track, input_root))
    return dict(drive(service, session, prompt), session_config=session['config'])


SPEC = dict(output_tokens=OUTPUT_TOKENS, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS, tracks=TRACKS)
