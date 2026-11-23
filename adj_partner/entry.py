"""Single command-line entry: ``adj <command>``.

Session commands (tui, new, run, resume, status, ...) go to the interactive harness CLI unchanged;
the namespaces below dispatch to the other tools.
"""

import sys

ENV_AWARE = {'task'}

NAMESPACES = {
    'task': ('adj_partner.orchestration.cli', 'TaskSpec workflows and campaigns: run, resume, status, export, campaign'),
    'library': ('adj_partner.library.__main__', 'project literature library: search, paper, related, enrich, expand, catalog'),
    'doctor': ('adj_partner.doctor', 'installation checks; `adj doctor runs` for a failure census'),
    'legacy': ('legacy.main', 'earlier single-process agent, kept for comparison'),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ('-h', '--help', 'help'):
        print('usage: adj <command> [...]\n\nnamespaces:')
        for name, (_, summary) in NAMESPACES.items():
            print(f'  {name:<8} {summary}')
        print('\nall other commands are session commands: adj tui | new | run | resume | status | sessions | ...'
              '\n(see `adj sessions --help`, `adj run --help`)')
        return 0
    if (not argv or argv[0] not in NAMESPACES or argv[0] in ENV_AWARE) and not _has_env_choice(argv):
        from .application_config import host_env_file
        selected = host_env_file()
        if selected:
            argv = [argv[0], '--env-file', str(selected), *argv[1:]] if argv and argv[0] in ENV_AWARE \
                else ['--env-file', str(selected), *argv]
    if argv and argv[0] in NAMESPACES:
        module = __import__(NAMESPACES[argv[0]][0], fromlist=['main'])
        if argv[0] == 'legacy':
            sys.argv = ['adj legacy', *argv[1:]]
            return module.main() or 0
        return module.main(argv[1:])
    from .cli import main as session_main
    return session_main(argv)


def _has_env_choice(argv):
    return any(a == '--env-file' or a.startswith('--env-file=') or a == '--offline-demo' for a in argv)


if __name__ == '__main__':
    raise SystemExit(main())
