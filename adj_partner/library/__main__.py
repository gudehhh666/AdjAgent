"""python -m adj_partner.library {stats|search|paper|catalog} — read the project literature library."""

import argparse
import json
import sys

from . import shared


def _enricher(library):
    from ..application_config import PROJECT_ROOT, host_env_file
    from ..native.scholarly import ScholarlyGateway
    from .enrich import Enricher, Fetcher
    try:
        pubmed = ScholarlyGateway.from_environment(PROJECT_ROOT, host_env_file()).pubmed
        key, email = pubmed.api_key, pubmed.email
    except ValueError:
        key = email = None
    log = lambda message: print(message, file=sys.stderr, flush=True)
    return Enricher(library, Fetcher(ncbi_key=key, email=email), log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m adj_partner.library')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('stats')
    search = sub.add_parser('search')
    search.add_argument('query')
    search.add_argument('--limit', type=int, default=10)
    search.add_argument('--adjuvant')
    search.add_argument('--project')
    search.add_argument('--min-text', choices=['title_only', 'abstract', 'full_text'])
    search.add_argument('--max-year', type=int)
    paper = sub.add_parser('paper')
    paper.add_argument('paper_id')
    enrich = sub.add_parser('enrich', help='fetch open full text and missing abstracts for filed papers')
    enrich.add_argument('--abstract-limit', type=int)
    enrich.add_argument('--fulltext-limit', type=int)
    expand = sub.add_parser('expand', help='follow references, citations and similar articles from adjuvant full texts')
    expand.add_argument('--per-seed', type=int, default=20)
    expand.add_argument('--max-new', type=int, default=2000)
    expand.add_argument('--limit-seeds', type=int)
    sub.add_parser('related').add_argument('paper_id')
    catalog = sub.add_parser('catalog')
    catalog.add_argument('--write', action='store_true', help='write CATALOG.md next to the library')
    args = parser.parse_args(argv)
    library = shared()
    if library is None:
        parser.error('the literature library is disabled (ADJ_LITERATURE_LIBRARY=off)')
    if args.command == 'stats':
        result = library.stats()
    elif args.command == 'search':
        result = library.search(args.query, limit=args.limit, adjuvant=args.adjuvant, project=args.project,
                                min_text=args.min_text, max_year=args.max_year)
    elif args.command == 'paper':
        result = library.paper(args.paper_id)
    elif args.command == 'related':
        paper = library.paper(args.paper_id)
        result = library.related(paper['paper_id']) if paper else None
    elif args.command in ('enrich', 'expand'):
        enricher = _enricher(library)
        if args.command == 'enrich':
            result = {'abstracts': enricher.abstracts(limit=args.abstract_limit),
                      'fulltext': enricher.fulltext(limit=args.fulltext_limit)}
        else:
            result = enricher.expand(per_seed=args.per_seed, max_new=args.max_new, limit_seeds=args.limit_seeds)
            result['fulltext'] = enricher.fulltext()
        library.retag()
        (library.root / 'CATALOG.md').write_text(library.catalog(), encoding='utf-8')
    else:
        text = library.catalog()
        if args.write:
            (library.root / 'CATALOG.md').write_text(text, encoding='utf-8')
            print(library.root / 'CATALOG.md')
        else:
            sys.stdout.write(text)
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
