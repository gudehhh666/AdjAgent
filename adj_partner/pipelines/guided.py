"""F6 guided evidence-bound session: host guidance in the session instructions, the user prompt unchanged.

The guidance is case material supplied by the caller. Guidance that names or steers toward a target answer is
only allowed in runs labelled controlled; ``FROZEN_GUIDANCE_SHA256`` pins the text used by the frozen PLGA case.
"""
import hashlib

from .sessions import drive

PROFILE = 'research-evidence-bound'
BUDGET = {'max_steps': 16, 'max_input_tokens': 150000, 'max_output_tokens': 16000, 'wall_time_seconds': 900.0}
REQUEST_OUTPUT_TOKENS, REQUEST_TIMEOUT_SECONDS = 8000, 180.0
FROZEN_GUIDANCE_SHA256 = {'plga-structure-v2': '31f5fdee4fc3cf38f13c1cb3fddf6a2a635194b16911d55417e35fb356faca8e'}


def service_options():
    return dict(request_output_tokens=REQUEST_OUTPUT_TOKENS, request_timeout_seconds=REQUEST_TIMEOUT_SECONDS)


def run(service, prompt, guidance, *, model=None):
    """One fresh session; the answer is the reply content, else the delivery."""
    session = service.create_session(profile=PROFILE, instructions=guidance, allow_network=True, allowed_roots=[],
                                     budget=BUDGET)
    if model:
        service.set_model(session['id'], model)
    out = drive(service, session, prompt)
    state = out.get('runtime', {}).get('state', {})
    out['content'] = (state.get('reply') or {}).get('content') or out['content']
    out['guidance_sha256'] = hashlib.sha256(guidance.encode()).hexdigest()
    return out


SPEC = dict(profile=PROFILE, budget=BUDGET, request_output_tokens=REQUEST_OUTPUT_TOKENS,
            request_timeout_seconds=REQUEST_TIMEOUT_SECONDS, frozen_guidance_sha256=FROZEN_GUIDANCE_SHA256)
