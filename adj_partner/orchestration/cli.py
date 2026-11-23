"""Command-line adapter for the unified framework; no independent model loop."""
import argparse
import json
from pathlib import Path
import sys
from ..application import AgentService
from .contracts import TaskSpec
from .controller import AgentFramework
from .campaign import Campaign, resume_issue
from .exports import export_task


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', default='.runtime/unified-agent', help='Persistent AgentService state directory')
    p.add_argument('--env-file', help='Explicit host environment file; never discovered automatically')
    p.add_argument('--project-root', help='Host project root for provider and data configuration')
    p.add_argument('--dataset-dir', help='Explicit host dataset directory')
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Submit and execute an explicit TaskSpec')
    run.add_argument('--spec', required=True, help='Path to TaskSpec JSON')
    resume = sub.add_parser('resume', help='Resume persisted task without replacing its identity')
    resume.add_argument('fingerprint')
    status = sub.add_parser('status', help='Inspect a task, campaign, or all tasks and campaigns')
    status.add_argument('fingerprint', nargs='?')
    export = sub.add_parser('export', help='Export verified delivery and status records')
    export.add_argument('fingerprint'); export.add_argument('--output', required=True)
    campaign = sub.add_parser('campaign', help='Prepare a persistent public-question queue')
    campaign.add_argument('--dataset', required=True)
    campaign.add_argument('--task-type', choices=['qa', 'design'], required=True)
    campaign.add_argument('--limit', type=int)
    campaign.add_argument('--offset', type=int, default=0)
    campaign.add_argument('--model', default='gpt-5.6-luna')
    campaign.add_argument('--budget', type=json.loads, default={}, help='JSON object of budget overrides')
    campaign.add_argument('--execute', action='store_true', help='Execute queued rows serially; default is queue only')
    campaign.add_argument('--retry-failed', action='store_true',
                          help='With --execute: re-queue rows that failed for transient provider reasons (bounded)')
    resume_campaign = sub.add_parser('campaign-run', help='Resume an existing campaign by id: reclaims rows left running by a dead process')
    resume_campaign.add_argument('campaign_id')
    resume_campaign.add_argument('--retry-failed', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    service = None
    try:
        spec = TaskSpec.from_dict(json.loads(Path(args.spec).expanduser().read_text())) if args.command == 'run' else None
        host_options = {key: getattr(args, key) for key in ('env_file', 'project_root', 'dataset_dir')
                        if getattr(args, key) is not None}
        service = AgentService(Path(args.state).expanduser().resolve(), **host_options)
        framework = AgentFramework(service)
        if args.command == 'run':
            item = framework.submit(spec)
            issue = resume_issue(framework, item)
            if issue: raise RuntimeError(issue)
            value = framework.run(item['fingerprint'])
        elif args.command == 'resume':
            item = framework.repository.get(args.fingerprint)
            issue = resume_issue(framework, item)
            if issue: raise RuntimeError(issue)
            value = framework.run(args.fingerprint)
        elif args.command == 'status':
            campaigns = Campaign(framework)
            if args.fingerprint:
                value = framework.repository.find(args.fingerprint)
                if value is None: value = campaigns.get(args.fingerprint)
            else:
                value = {'tasks': framework.repository.list(), 'campaigns': campaigns.list()}
        elif args.command == 'export':
            value = export_task(framework, args.fingerprint, args.output)
        elif args.command == 'campaign-run':
            value = Campaign(framework).execute(args.campaign_id, retry_failed=args.retry_failed)
        else:
            if not isinstance(args.budget, dict): raise ValueError('Budget must be a JSON object')
            campaigns = Campaign(framework)
            value = campaigns.prepare(args.dataset, task_type=args.task_type, limit=args.limit,
                                      offset=args.offset, model=args.model, budget=args.budget)
            if args.execute: value = campaigns.execute(value['id'], retry_failed=args.retry_failed)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if args.command in {'run', 'resume'} and value.get('execution_status') != 'succeeded':
            return 1
        if args.command in {'campaign', 'campaign-run'} and any(i['status'] not in {'queued', 'succeeded'} for i in value['items']):
            return 1
        return 0
    except Exception as exc:
        message = str(exc)
        if service is not None: message = service.store.redact(message)
        print(json.dumps({'error': {'code': type(exc).__name__, 'message': message}}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        if service is not None: service.close()


if __name__ == '__main__':
    raise SystemExit(main())
