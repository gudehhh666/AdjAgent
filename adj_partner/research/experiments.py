"""Evidence-bound, reported experimental feedback; no automatic scientific certification."""
from .checkpoints import LEVELS
from .validation import digest, identifier, ids, text


def contract_hash(card):
    # Updating the evidence narrative does not change an assay's contract.
    # Changed hypotheses, readouts, context or outcome rules invalidate reuse.
    return digest({k:v for k,v in card.items() if k not in {'current_evidence','claim_ids'}})


def _card(design, checkpoint_id):
    for card in design['verification'].get('checkpoints',[]):
        if card['id']==checkpoint_id:return card
    raise ValueError('Unknown checkpoint in target design')


class ExperimentRecords:
    def record_experiment_result(self, session_id, run_id, *, result_key, design_id, checkpoint_id,
                                 evidence_ids, measurement_level, quality, context_status, outcome,
                                 interpretation, context, reported_by, supersedes=None):
        """Host-only import. Quality/context/outcome are attributed reporter judgments.

        No model-callable source-create or experiment-result-create tool exists.
        Files establish what was reported, not whether an assay was performed correctly.
        """
        identifier(result_key);identifier(checkpoint_id);ids(evidence_ids)
        if supersedes is not None:identifier(supersedes)
        if not evidence_ids:raise ValueError('Experimental result requires source evidence')
        for key,value in [('interpretation',interpretation),('context',context),('reported_by',reported_by)]:text(value,key,4000)
        if measurement_level not in LEVELS:raise ValueError('Unknown measurement level')
        if quality not in ('pass','fail','unknown'):raise ValueError('Unknown reported quality')
        if context_status not in ('matched','mismatched','unknown'):raise ValueError('Unknown reported context match')
        if outcome not in ('supports','opposes','inconclusive'):raise ValueError('Unknown reported outcome')
        def build(conn,tid,bindings):
            target=self._design_bind(conn,session_id,tid,'design_revision',design_id,bindings)
            card=_card(target['payload']['design'],checkpoint_id)
            if supersedes is not None:
                old=self._design_bind(conn,session_id,tid,'experiment_result',supersedes,bindings,inherit=False)['payload']
                if (old['design_id']!=design_id or old['checkpoint_id']!=checkpoint_id or
                        old['checkpoint_contract_hash']!=contract_hash(card)):
                    raise ValueError('Correction must target the same checkpoint and design contract')
            for eid in evidence_ids:
                evidence=self._design_bind(conn,session_id,tid,'evidence',eid,bindings)
                src=self._get(conn,'source',session_id,tid,evidence['source_id'])
                if src['kind'] not in ('userfile','computed') or src['provenance'].get('experiment_data') is not True:
                    raise ValueError('Result must bind host-ingested experimental source material')
            reasons=[]
            if quality!='pass':reasons.append('assay_quality_not_passed')
            if context_status!='matched':reasons.append('experimental_context_not_matched')
            if card.get('decision_level')!=measurement_level:reasons.append('measurement_level_does_not_resolve_checkpoint')
            return dict(result_key=result_key,design_id=design_id,checkpoint_id=checkpoint_id,
                checkpoint_contract_hash=contract_hash(card),branch=card['branch'],evidence_ids=evidence_ids,
                measurement_level=measurement_level,quality=quality,context_status=context_status,outcome=outcome,
                effective_outcome='inconclusive' if reasons else outcome,inconclusive_reasons=reasons,
                interpretation=interpretation,context=context,reported_by=reported_by,supersedes=supersedes,
                scientific_validation='reported_interpretations_not_independently_validated')
        return self._create_design_record(session_id,run_id,'experiment_result',build,identity_fields=('result_key',))

    def experiment_state(self,session_id,run_id,*,design_id):
        target=self.get_design_record(session_id,run_id,design_id)
        if target['kind']!='design_revision':raise ValueError('Expected design revision')
        issues=self.validate_design_record(session_id,run_id,design_id)
        if issues:raise ValueError('Stale design bindings: '+'; '.join(issues))
        ancestors={design_id};cursor=target['payload']['parent_id']
        while cursor:
            if cursor in ancestors:raise ValueError('Cyclic design lineage')
            ancestors.add(cursor);cursor=self.get_design_record(session_id,run_id,cursor)['payload']['parent_id']
        results=self.list_design_records(session_id,run_id,'experiment_result');rows=[]
        for card in target['payload']['design']['verification'].get('checkpoints',[]):
            candidates=[r for r in results if r['payload']['design_id'] in ancestors and
                r['payload']['checkpoint_id']==card['id'] and r['payload']['checkpoint_contract_hash']==contract_hash(card)]
            superseded={r['payload'].get('supersedes') for r in candidates}-{None}
            candidates=[r for r in candidates if r['id'] not in superseded]
            outcomes=set();valid=[];stale=[]
            for result in candidates:
                problems=self.validate_design_record(session_id,run_id,result['id'])
                if problems:stale.append({'result_id':result['id'],'issues':problems});continue
                valid.append(result['id']);o=result['payload']['effective_outcome']
                if o!='inconclusive':outcomes.add(o)
            status=('conflicting' if len(outcomes)>1 else next(iter(outcomes)) if outcomes else
                    'inconclusive' if candidates else 'unmeasured')
            rows.append({'checkpoint_id':card['id'],'branch':card['branch'],'decision_level':card.get('decision_level'),
                         'status':status,'result_ids':valid,'stale_results':stale,
                         'next_step':card[status if status in ('supports','opposes') else 'inconclusive'],
                         'next_step_scope':'Selected from the recorded plan; not a new scientific inference or executed experiment.',
                         'superseded_result_ids':sorted(superseded),
                         'reported_results':[r['payload'] for r in candidates if r['id'] in valid]})
        function={r['status'] for r in rows if r['branch']=='function'}
        mechanism={r['status'] for r in rows if r['branch']=='mechanism'}
        if 'conflicting' in function or {'supports','opposes'}<=function:decision='resolve_functional_conflict'
        elif function=={'supports'}:
            decision='retain_function_reconsider_mechanism' if 'opposes' in mechanism or 'conflicting' in mechanism else 'retain_reported_function_mechanism_separate'
        elif function and function<={'opposes'}:decision='do_not_advance_under_tested_conditions'
        else:decision='functional_benefit_unresolved'
        state={'design_id':design_id,'checkpoints':rows,'decision':decision,
                'scientific_validation':'reported_interpretations_not_independently_validated',
                'limitations':['No overall causal mechanism or vaccine efficacy certification. Different evidence levels are not interchangeable.',
                               'Quality and context match are reporter declarations; host verifies source bytes and bindings only.']}
        verification=target['payload']['design']['verification']
        if 'mechanism_graph' in verification:
            from .mechanism_graph import analyze
            state['mechanism_graph']=analyze(verification['mechanism_graph'],verification['checkpoints'],
                                             {r['checkpoint_id']:r['status'] for r in rows},
                                             verification.get('literature_edges'))
        return state

    def _bind_experiment_results(self,conn,sid,tid,design,parent,ancestors,result_ids,bindings):
        if parent is None:raise ValueError('Experimental update requires a parent design')
        claims={c['id']:c for c in self._list(conn,'claim',sid,tid)}
        for rid in result_ids:
            result=self._design_bind(conn,sid,tid,'experiment_result',rid,bindings)
            p=result['payload']
            if p['design_id'] not in ancestors:raise ValueError('Experimental result must target an ancestor design')
            previous=_card(parent['payload']['design'],p['checkpoint_id'])
            updated=_card(design,p['checkpoint_id'])
            if previous['current_evidence']==updated['current_evidence']:
                raise ValueError('Affected checkpoint evidence narrative must have changed')
            linked={eid for cid in updated['claim_ids'] for eid in claims[cid]['evidence_ids']}
            if not set(p['evidence_ids'])<=linked:
                raise ValueError('Updated checkpoint must cite every applied experimental evidence selection')

    def revise_design_from_experiments(self,session_id,run_id,*,design_id,result_ids,design):
        ids(result_ids)
        if not result_ids:raise ValueError('Experimental update requires result IDs')
        parent=self.get_design_record(session_id,run_id,design_id)
        if parent['kind']!='design_revision':raise ValueError('Expected design revision')
        return self.create_design_revision(session_id,run_id,brief_id=parent['payload']['brief_id'],
            parent_id=design_id,design=design,result_ids=result_ids)
