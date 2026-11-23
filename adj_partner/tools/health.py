"""Run-local availability derived from completed adapter outcomes, never source text."""


def suspended_tools(store, run_id):
    for action in store.list_actions(run_id):
        result = action.get('result') or {}
        error = result.get('error') or {}
        if (action.get('finished_at') is not None
                and action.get('tool_name') == 'web_search'
                and result.get('status') == 'error'
                and error.get('code') == 'web_search_quota'):
            return {'web_search': 'web_search_quota'}
    return {}
