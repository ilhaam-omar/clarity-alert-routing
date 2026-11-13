"""Conditional routing experiment with a shared strict suppression gate.

The static arm deliberately shares the corrected policy's suppression eligibility
and non-resource fallback. It is not an optimisation-free comparator. Only
initial non-suppression routing and capacity priority are varied.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ACTIONS = ('suppress', 'passive', 'alert', 'escalate')
WINNERS = ('raw_winner_safety_priority', 'raw_winner_workflow_balanced', 'raw_winner_burden_priority')
FALSE = {'0', 'false', 'no', 'n', 'f'}
TRUE = {'1', 'true', 'yes', 'y', 't'}
SAFETY = {'standard', 'no_suppress', 'must_review_or_interrupt', 'must_interrupt'}
ORDER = {'suppress': ('suppress','passive','escalate','alert'),
         'passive': ('passive','suppress','escalate','alert'),
         'alert': ('alert','escalate','passive','suppress'),
         'escalate': ('escalate','alert','passive','suppress')}

def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing: raise ValueError(label + ' missing columns: ' + ', '.join(sorted(missing)))

def allowed_actions(value):
    if pd.isna(value): raise ValueError('Missing admissible action set')
    result = tuple(x.strip() for x in str(value).split('|'))
    if not result or any(x not in ACTIONS for x in result) or len(result) != len(set(result)):
        raise ValueError('Malformed admissible action set')
    return result

def booleans(values):
    text = values.astype(str).str.strip().str.lower()
    if not text.isin(TRUE | FALSE).all(): raise ValueError('Missing or malformed Boolean input')
    return text.isin(TRUE)

def shared_suppression_gate(decisions, threshold=0.8):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Suppression threshold must be between zero and one')
    require(decisions, ['admissible_actions','safety_class','material_change_since_prior',
                        'consensus_action','action_support_suppress_min',*WINNERS], 'Decisions')
    support = pd.to_numeric(decisions['action_support_suppress_min'], errors='raise')
    if not (np.isfinite(support) & support.between(0,1)).all():
        raise ValueError('Suppression support must be finite and between zero and one')
    if not decisions.safety_class.isin(SAFETY).all(): raise ValueError('Unknown safety class')
    changed = booleans(decisions.material_change_since_prior)
    allowed = decisions.admissible_actions.map(allowed_actions)
    gate = support.ge(threshold) & decisions.consensus_action.eq('suppress')
    for name in WINNERS: gate &= decisions[name].eq('suppress')
    gate &= decisions.safety_class.eq('standard') & ~changed
    gate &= allowed.map(lambda x: 'suppress' in x)
    return gate

def static_actions(decisions, gate, severity_threshold=8.0):
    """Apply only the declared matrix and common hard restrictions."""
    require(decisions, ['base_severity','certainty_class','safety_class','admissible_actions'], 'Decisions')
    if not gate.index.equals(decisions.index): raise ValueError('Suppression gate rows are misaligned')
    if not np.isfinite(severity_threshold): raise ValueError('Non-finite severity threshold')
    actions=[]; raw=[]
    for (_, row), eligible in zip(decisions.iterrows(), gate):
        severity=float(row.base_severity)
        if not np.isfinite(severity): raise ValueError('Non-finite severity')
        high=severity>=severity_threshold
        certain=str(row.certainty_class).lower()=='high'
        preferred=('alert' if certain else 'escalate') if high else ('passive' if certain else 'suppress')
        raw.append(preferred)
        allowed=set(allowed_actions(row.admissible_actions))
        if not eligible: allowed.discard('suppress')
        safety=str(row.safety_class)
        if safety=='must_interrupt':
            if 'alert' not in allowed: raise ValueError('Protected alert is not admissible')
            selected='alert'
        elif safety=='must_review_or_interrupt':
            preferred='alert' if certain else 'escalate'
            candidates=[x for x in (preferred,'alert','escalate') if x in allowed]
            if not candidates: raise ValueError('Protected review has no resource route')
            selected=candidates[0]
        else:
            if safety=='no_suppress': allowed.discard('suppress')
            candidates=[x for x in ORDER[preferred] if x in allowed]
            if not candidates: raise ValueError('No static route satisfies common safeguards')
            selected=candidates[0]
        actions.append(selected)
    return pd.Series(actions,index=decisions.index),pd.Series(raw,index=decisions.index)

def validate_signals(frame, gate):
    action=frame.pre_capacity_action.astype(str)
    allowed=frame.admissible_actions.map(allowed_actions)
    fallback=frame.fallback_non_resource_action.astype(str)
    # Protected signals may retain a placeholder fallback that can never be used.
    standard=~frame.safety_class.isin(['must_interrupt','must_review_or_interrupt'])
    checks={
        'unique_signal_ids':not frame.signal_id.duplicated().any(),
        'defined_routes':bool(action.isin(ACTIONS).all()),
        'fallbacks_are_non_resource_routes':bool(fallback.isin(['suppress','passive']).all()),
        'selected_routes_admissible':all(a in b for a,b in zip(action,allowed)),
        'suppression_gate_holds':bool(gate[action.eq('suppress')].all()),
        'suppression_fallback_gate_holds':bool(gate[fallback.eq('suppress')].all()),
        'nonprotected_fallback_admissible':all(a in b for a,b in zip(fallback[standard],allowed[standard])),
        'protected_alerts_retained':bool(action[frame.safety_class.eq('must_interrupt')].eq('alert').all()),
        'protected_reviews_retained':bool(action[frame.safety_class.eq('must_review_or_interrupt')].isin(['alert','escalate']).all())}
    failures=[k for k,v in checks.items() if not v]
    if failures: raise AssertionError('Signal checks failed: '+', '.join(failures))
    return checks

def build_policy_bundles(decisions, build_order_bundles, threshold=0.8, severity_threshold=8.0):
    gate=shared_suppression_gate(decisions,threshold)
    static=decisions.copy()
    static['pre_capacity_action'],static['matrix_preferred_action']=static_actions(decisions,gate,severity_threshold)
    checks={'clarity':validate_signals(decisions,gate),'static_common_gate':validate_signals(static,gate)}
    full=build_order_bundles(decisions)
    simpler=build_order_bundles(static)
    eligible_by_bundle=gate.groupby(decisions.bundle_id).all()
    for frame in (full,simpler):
        frame['bundle_suppression_eligible']=frame.bundle_id.map(eligible_by_bundle)
        if frame.bundle_suppression_eligible.isna().any(): raise AssertionError('Unmatched bundle suppression gate')
    # Explicitly share corrected fallbacks and ranking inputs; severity arms ignore the deltas.
    common=['safety_class','admissible_actions','protected_alert','protected_review',
            'fallback_non_resource_action','delta_frontline','delta_expert','highest_base_severity',
            'bundle_suppression_eligible']
    one=full.set_index('bundle_id');two=simpler.set_index('bundle_id')
    if set(one.index)!=set(two.index): raise AssertionError('Policies have different bundle denominators')
    for col in common:
        if not one[col].sort_index().equals(two[col].sort_index()):
            raise AssertionError('Common bundle field differs: '+col)
    return {'clarity':full,'static':simpler},checks,static
