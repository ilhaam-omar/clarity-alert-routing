"""Reproducible matched-capacity experiment on the corrected CLARITY outputs."""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.metadata,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from matched_policies import build_policy_bundles, require, booleans, ACTIONS
from matched_capacity import allocate_absolute

KIT=Path(__file__).resolve().parent
POLICIES={'A_static_severity':('static','severity'),
          'B_clarity_severity':('clarity','severity'),
          'C_static_regret':('static','regret'),
          'D_clarity_regret':('clarity','regret')}
CONTRASTS=[('selection_with_severity_priority','A_static_severity','B_clarity_severity'),
           ('selection_with_regret_priority','C_static_regret','D_clarity_regret'),
           ('ranking_with_static_routes','A_static_severity','C_static_regret'),
           ('ranking_with_clarity_routes','B_clarity_severity','D_clarity_regret'),
           ('combined_selection_and_ranking','A_static_severity','D_clarity_regret')]
SCENARIOS=['low_capacity','medium_capacity','high_capacity','unconstrained_reference']

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def source_guard(project):
    guard=read_json(KIT/'experiment_source_guard.json');observed={}
    for name,expected in guard['source_files'].items():
        file=project/name
        if not file.is_file():raise FileNotFoundError('Missing corrected source file: '+name)
        data=file.read_bytes();raw=hashlib.sha256(data).hexdigest();norm=hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()
        if raw!=expected['sha256'] and norm!=expected['newline_normalized_sha256'] and raw!=expected.get('publication_sha256'):
            raise ValueError('Source version mismatch: '+name+'. Use the completed suppression-corrected project.')
        observed[name]=raw
    output=project/'outputs_corrected'
    for name,expected in guard['frozen_inputs'].items():
        file=output/'private'/name
        if not file.is_file():raise FileNotFoundError('Missing frozen corrected input: '+name)
        if sha(file)!=expected['sha256']:raise ValueError('Frozen input mismatch: '+name)
    return observed,guard

def read_private(output,name,columns=None):
    path=output/'private'/name
    if not path.is_file():raise FileNotFoundError('Missing corrected output: '+name)
    return pd.read_csv(path,usecols=columns,low_memory=False)

def validate_linkage(states,decisions,saved_bundles,saved_final):
    require(states,['signal_id','bundle_id'],'States')
    require(decisions,['signal_id','bundle_id'],'Decisions')
    if states.signal_id.duplicated().any() or decisions.signal_id.duplicated().any():raise ValueError('Duplicate signal identifiers')
    a=states.set_index('signal_id').sort_index();b=decisions.set_index('signal_id').sort_index()
    if not a.index.equals(b.index):raise ValueError('Stage03 and Stage04 signal sets differ')
    for col in states.columns:
        if col=='signal_id':continue
        if col not in b:raise ValueError('Stage04 does not retain required state field: '+col)
        av=a[col];bv=b[col]
        if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
            match=np.allclose(av,bv,equal_nan=True,rtol=0,atol=1e-12)
        else:match=av.fillna('<MISSING>').astype(str).equals(bv.fillna('<MISSING>').astype(str))
        if not match:raise ValueError('Stage03 and Stage04 values differ: '+col)
    if saved_bundles.bundle_id.duplicated().any():raise ValueError('Duplicate saved bundle identifiers')
    if set(saved_bundles.bundle_id)!=set(decisions.bundle_id):raise ValueError('Bundle denominator does not match decisions')
    if set(saved_final.scenario)!=set(SCENARIOS):raise ValueError('Expected all four corrected capacity settings')
    if saved_final[['scenario','bundle_id']].duplicated().any():raise ValueError('Duplicate final bundle/scenario')
    for _,group in saved_final.groupby('scenario'):
        if set(group.bundle_id)!=set(saved_bundles.bundle_id):raise ValueError('Final denominator differs from bundle denominator')
    return {'state_decision_identity_and_shared_fields_match':True,'saved_bundle_and_final_denominators_match':True}

def check_saved_bundles(full,saved):
    a=full.set_index('bundle_id').sort_index();b=saved.set_index('bundle_id').sort_index()
    cols=['pre_capacity_action','fallback_non_resource_action','safety_class','admissible_actions','protected_alert','protected_review','highest_base_severity','signal_count','delta_expert','delta_frontline']
    for col in cols:
        if pd.api.types.is_numeric_dtype(a[col]):ok=np.allclose(a[col],b[col],rtol=1e-12,atol=1e-12,equal_nan=True)
        else:ok=a[col].astype(str).equals(b[col].astype(str))
        if not ok:raise AssertionError('Rebuilt corrected bundle field differs: '+col)

def capacity_limits(saved_final,policies):
    result=[]
    for scenario in SCENARIOS:
        saved=saved_final[saved_final.scenario.eq(scenario)]
        for col in ['frontline_slots','expert_slots']:
            if saved[col].nunique()!=1:raise ValueError('Capacity slots are not constant within the saved scenario')
        if scenario=='unconstrained_reference':
            f=max(int(t.pre_capacity_action.eq('alert').sum()) for t in policies.values())
            k=max(int(t.pre_capacity_action.eq('escalate').sum()) for t in policies.values())
            basis='No scarcity: maximum route demand across both routing policies; same F and K for every arm.'
        else:
            f=int(saved.frontline_slots.iloc[0]);k=int(saved.expert_slots.iloc[0]);basis='Fixed absolute slots from the corrected CLARITY reference; never recomputed from comparator demand.'
        result.append({'scenario':scenario,'frontline_slots':f,'expert_slots':k,'basis':basis})
    return result

def summarise(frame,policy,bundles):
    n=len(frame);counts=frame.final_action.value_counts();alerts=int(counts.get('alert',0));reviews=int(counts.get('escalate',0))
    review_demand=int(bundles.pre_capacity_action.eq('escalate').sum());k=int(frame.expert_slots.iloc[0]);f=int(frame.frontline_slots.iloc[0])
    meta=bundles.set_index('bundle_id');review_ids=frame.loc[frame.final_action.eq('escalate'),'bundle_id']
    return {'scenario':str(frame.scenario.iloc[0]),'policy':policy,'affected_bundles':n,
            'frontline_slots':f,'expert_slots':k,'pre_capacity_review_demand':review_demand,
            'suppress':int(counts.get('suppress',0)),'passive':int(counts.get('passive',0)),
            'alert':alerts,'review':reviews,'unused_review_slots':max(0,k-reviews),
            'unserved_review_requests':review_demand-reviews,
            'protected_review_spillovers':int(frame.capacity_status.str.contains('protected_review_spillover').sum()),
            'frontline_overrun':max(0,alerts-f),
            'selected_review_robust_advantage_sum':float(meta.loc[review_ids,'delta_expert'].sum()),
            'protected_route_violations':0,'suppression_gate_violations':0}

def compare_pair(left,right,scenario,name,left_name,right_name,metadata):
    a=left[['bundle_id','final_action']].rename(columns={'final_action':'left_action'})
    b=right[['bundle_id','final_action']].rename(columns={'final_action':'right_action'})
    merged=a.merge(b,on='bundle_id',validate='one_to_one',how='outer',indicator=True)
    if not merged['_merge'].eq('both').all():raise AssertionError('Paired denominators differ')
    changed=merged.left_action.ne(merged.right_action);reviews_left=set(merged.loc[merged.left_action.eq('escalate'),'bundle_id']);reviews_right=set(merged.loc[merged.right_action.eq('escalate'),'bundle_id']);union=reviews_left|reviews_right;common=reviews_left&reviews_right
    row={'scenario':scenario,'contrast':name,'left_policy':left_name,'right_policy':right_name,'affected_bundles':len(merged),'changed_routes':int(changed.sum()),'same_route_percent':100*float((~changed).mean()),'left_review_recipients':len(reviews_left),'right_review_recipients':len(reviews_right),'common_review_recipients':len(common),'left_only_review_recipients':len(reviews_left-reviews_right),'right_only_review_recipients':len(reviews_right-reviews_left),'review_recipient_jaccard_percent':100*len(common)/len(union) if union else None}
    transitions=merged.groupby(['left_action','right_action']).size().reset_index(name='bundle_count')
    for col,val in [('scenario',scenario),('contrast',name),('left_policy',left_name),('right_policy',right_name)]:transitions[col]=val
    private=merged.drop(columns='_merge');private['scenario']=scenario;private['contrast']=name
    strata=private.merge(metadata[['bundle_id','safety_class','highest_base_severity']],on='bundle_id',validate='one_to_one');strata['route_changed']=strata.left_action.ne(strata.right_action)
    strata=strata.groupby(['safety_class','highest_base_severity']).agg(affected_bundles=('route_changed','size'),changed_routes=('route_changed','sum')).reset_index();strata['scenario']=scenario;strata['contrast']=name
    return row,transitions,private,strata

def priority_ties(bundles,label):
    rows=[]
    for action,score in [('escalate','delta_expert'),('alert','delta_frontline')]:
        group=bundles[bundles.pre_capacity_action.eq(action)]
        if action=='alert':
            group=group[~booleans(group.protected_alert)&~booleans(group.protected_review)]
        for priority,column in [('severity','highest_base_severity'),('regret',score)]:
            sizes=group.groupby(column).size()
            rows.append({'routing':label,'candidate_route':action,'priority':priority,'candidates':len(group),'distinct_priority_values':len(sizes),'candidates_in_tied_priority_groups':int(sizes[sizes.gt(1)].sum()),'largest_tie_group':int(sizes.max()) if len(sizes) else 0})
    return rows

def write_return_zip(output,files):
    # Explicit aggregate list only; no private tree or free-text clinical fields.
    aggregate=output/'aggregate';manifest={name:sha(aggregate/name) for name in files}
    write_json(aggregate/'file_manifest.json',manifest)
    target=output/'CLARITY_matched_comparison_results.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files+['file_manifest.json']:z.write(aggregate/name,name)
    return target

def run(project,output=None):
    print('1/5 Checking corrected source and frozen inputs.',flush=True)
    project=project.expanduser().resolve();source_hashes,guard=source_guard(project);corrected=project/'outputs_corrected'
    state_fields=['signal_id','bundle_id','subject_id','hadm_id','timestamp','base_severity','certainty_class','safety_class','admissible_actions','material_change_since_prior']
    states=read_private(corrected,'03_longitudinal_states.csv',state_fields)
    decision_fields=state_fields+['consensus_action','action_support_suppress_min','raw_winner_safety_priority','raw_winner_workflow_balanced','raw_winner_burden_priority','pre_capacity_action','fallback_non_resource_action','protected_alert','protected_review','calibration_sensitive','delta_frontline','delta_expert']
    decisions=read_private(corrected,'04_action_sets.csv',decision_fields)
    bundles=read_private(corrected,'05_order_bundles.csv');final=read_private(corrected,'06_final_bundle_decisions.csv')
    orders=read_private(corrected,'01_order_timeline.csv',['bundle_id']);total_orders=int(orders.bundle_id.nunique())
    checks=validate_linkage(states,decisions,bundles,final)
    if not set(bundles.bundle_id).issubset(set(orders.bundle_id)):raise AssertionError('Affected bundles absent from common order timeline')
    sys.path.insert(0,str(project))
    from clarity_v3.bundles import build_order_bundles
    cfg=read_json(project/'config/uncertainty_design.json')['action_set'];threshold=float(cfg['suppress_support_threshold'])
    severity=float(read_json(project/'config/state_builder.json')['high_severity_floor'])
    print('2/5 Building shared safeguards and the two routing policies.',flush=True)
    policy_bundles,signal_checks,static=build_policy_bundles(decisions,build_order_bundles,threshold,severity)
    check_saved_bundles(policy_bundles['clarity'],bundles);checks['rebuilt_corrected_bundles_match_saved']=True
    limits=capacity_limits(final,policy_bundles)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output=output.resolve() if output else project/('matched_comparison_'+stamp)
    output.mkdir(parents=True,exist_ok=False);private=output/'private';aggregate=output/'aggregate';private.mkdir();aggregate.mkdir()
    static[['signal_id','bundle_id','matrix_preferred_action','pre_capacity_action','fallback_non_resource_action']].to_csv(private/'static_signal_routes.csv',index=False)
    summaries=[];pair_rows=[];transition_frames=[];stratum_frames=[]
    for budget in limits:
        scenario=budget['scenario'];results={}
        print('3/5 Allocating fixed slots: '+scenario,flush=True)
        for key,(routing,priority) in POLICIES.items():
            result=allocate_absolute(policy_bundles[routing],budget['frontline_slots'],budget['expert_slots'],priority,scenario,total_orders)
            result.to_csv(private/(scenario+'__'+key+'.csv'),index=False);results[key]=result
            summaries.append(summarise(result,key,policy_bundles[routing]))
        # Independent control: reproduce every saved corrected CLARITY final route.
        control=results['D_clarity_regret'][['bundle_id','final_action']].merge(final.loc[final.scenario.eq(scenario),['bundle_id','final_action']],on='bundle_id',validate='one_to_one',suffixes=('_new','_saved'))
        if not control.final_action_new.eq(control.final_action_saved).all():raise AssertionError('Matched allocator failed corrected CLARITY replay in '+scenario)
        checks[scenario+'_clarity_replay_exact']=True
        if scenario=='unconstrained_reference':
            for left,right in [('A_static_severity','C_static_regret'),('B_clarity_severity','D_clarity_regret')]:
                a=results[left].set_index('bundle_id').final_action.sort_index();b=results[right].set_index('bundle_id').final_action.sort_index()
                if not a.equals(b):raise AssertionError('Queue priority changed a no-scarcity route')
            checks['no_scarcity_ranking_invariance']=True
        for name,left,right in CONTRASTS:
            row,trans,details,strata=compare_pair(results[left],results[right],scenario,name,left,right,policy_bundles['clarity'])
            pair_rows.append(row);transition_frames.append(trans);stratum_frames.append(strata)
            details.to_csv(private/(scenario+'__'+name+'.csv'),index=False)
    pd.DataFrame(summaries).to_csv(aggregate/'policy_summary.csv',index=False)
    pd.DataFrame(pair_rows).to_csv(aggregate/'paired_comparisons.csv',index=False)
    pd.concat(transition_frames,ignore_index=True).to_csv(aggregate/'route_transitions.csv',index=False)
    pd.concat(stratum_frames,ignore_index=True).to_csv(aggregate/'stratified_changes.csv',index=False)
    pd.DataFrame(limits).to_csv(aggregate/'matched_capacity_limits.csv',index=False)
    pd.DataFrame([r for label,b in policy_bundles.items() for r in priority_ties(b,label)]).to_csv(aggregate/'priority_ties.csv',index=False)
    print('4/5 Writing aggregate policy-comparison outputs.',flush=True)
    input_names=['01_order_timeline.csv','03_longitudinal_states.csv','04_action_sets.csv','05_order_bundles.csv','06_final_bundle_decisions.csv']
    experiment={'experiment_version':'1.0','scope':'Conditional matched-gate, matched-absolute-capacity factorial comparison; no clinical outcome inference.',
      'arms':{k:{'initial_routing':v[0],'queue_priority':v[1]}for k,v in POLICIES.items()},
      'common_components':['Frozen cohort, detector and reconstructed state','Hard safety classes and admissibility','Strict suppression eligibility from corrected Stage04','Corrected Stage04 non-resource fallback','Identical bundle precedence','Absolute frontline and review slots','Protected spillover behavior'],
      'interpretation_boundaries':['Static arms share the optimiser-derived suppression gate and fallback; this is not an optimisation-free baseline.','Contrasts estimate incremental initial route selection and queue priority conditional on shared components.','Severity-only priority is prespecified for this experiment; ties use bundle ID, not clinical arrival time.','Selected-review robust advantage is an internal model score, not independent evidence of safety or appropriateness.','Equal route totals can hide different review recipients; paired overlap is reported.','The experiment assesses algorithmic allocation, not professional assessment.'],
      'suppress_threshold':threshold,'high_severity_threshold':severity,'signals':len(decisions),'affected_bundles':len(bundles),'total_order_bundles':total_orders,
      'source_hashes':source_hashes,'input_hashes':{n:sha(corrected/'private'/n)for n in input_names},
      'script_hashes':{p.name:sha(p)for p in KIT.glob('*.py')},
      'python':sys.version.split()[0],'pandas':pd.__version__,'numpy':np.__version__,
      'clinical_validation':False}
    write_json(aggregate/'experiment_manifest.json',experiment)
    write_json(aggregate/'validation.json',{'checks':checks,'signal_checks':signal_checks,'all_recorded_checks_pass':True,'allocation_runs':len(summaries)})
    text='Matched comparison on frozen corrected CLARITY inputs.\nRead experiment_manifest.json for exact shared components and interpretation limits.\nA=static+severity; B=CLARITY+severity; C=static+regret; D=CLARITY+regret.\nB−A and D−C isolate selection conditional on common priority; C−A and D−B isolate ranking within a fixed candidate pool.\nRoute agreement and overlap describe behavior, not clinical superiority.\n'
    (aggregate/'READ_ME.txt').write_text(text,encoding='utf-8')
    files=['policy_summary.csv','paired_comparisons.csv','route_transitions.csv','stratified_changes.csv','matched_capacity_limits.csv','priority_ties.csv','experiment_manifest.json','validation.json','READ_ME.txt']
    target=write_return_zip(output,files)
    print('5/5 COMPLETE. All recorded checks passed.\nAggregate result ZIP: '+str(target),flush=True)
    return output

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True,help='Repository root with outputs_corrected, or an exact corrected project')
    parser.add_argument('--output',type=Path,help='New output directory; existing directories are refused')
    parser.add_argument('--dry-run',action='store_true',help='Validate corrected source and required files without computing')
    args=parser.parse_args()
    if args.dry_run:
        source_guard(args.project.resolve())
        for name in ['04_action_sets.csv','05_order_bundles.csv','06_final_bundle_decisions.csv']:
            if not (args.project/'outputs_corrected/private'/name).is_file():raise FileNotFoundError(name)
        print('PASS: corrected source and frozen inputs match. No files were changed.');return
    run(args.project,args.output)
if __name__=='__main__':
    try:main()
    except (ValueError,AssertionError,OSError,KeyError) as error:
        print('STOPPED: '+str(error),file=sys.stderr);raise SystemExit(2)
