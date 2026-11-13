from pathlib import Path
import json, sys, tempfile, unittest, zipfile
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kit'))
from matched_policies import shared_suppression_gate, static_actions, validate_signals, WINNERS
from run_matched_comparison import capacity_limits, write_return_zip, priority_ties, SCENARIOS

def signal(**changes):
    result={'signal_id':'synthetic','bundle_id':'synthetic','safety_class':'standard',
            'admissible_actions':'suppress|passive|alert|escalate',
            'material_change_since_prior':False,'consensus_action':'suppress',
            'action_support_suppress_min':.8,'base_severity':5,'certainty_class':'low',
            'pre_capacity_action':'suppress','fallback_non_resource_action':'suppress'}
    result.update({key:'suppress' for key in WINNERS});result.update(changes);return result

class PolicyExperimentTests(unittest.TestCase):
    def test_gate_requires_every_condition_and_accepts_inclusive_threshold(self):
        rows=[signal(),signal(action_support_suppress_min=.799999),
              signal(material_change_since_prior=True),signal(safety_class='no_suppress'),
              signal(consensus_action='passive'),signal(admissible_actions='passive|alert|escalate')]
        rows.extend(signal(**{key:'passive'}) for key in WINNERS)
        self.assertEqual(shared_suppression_gate(pd.DataFrame(rows)).tolist(),[True]+[False]*8)

    def test_unknown_material_change_does_not_silently_pass(self):
        for val in [None,'unknown']:
            with self.assertRaises(ValueError):shared_suppression_gate(pd.DataFrame([signal(material_change_since_prior=val)]))

    def test_matrix_boundary_and_suppression_gate(self):
        rows=[signal(base_severity=8,certainty_class='high'),signal(base_severity=8),
              signal(certainty_class='high'),signal(),signal(material_change_since_prior=True)]
        frame=pd.DataFrame(rows);actions,_=static_actions(frame,shared_suppression_gate(frame))
        self.assertEqual(actions.tolist(),['alert','escalate','passive','suppress','passive'])

    def test_hard_protection_overrides_matrix(self):
        rows=[signal(safety_class='must_interrupt',admissible_actions='alert'),
              signal(safety_class='must_review_or_interrupt',admissible_actions='alert|escalate')]
        frame=pd.DataFrame(rows);actions,_=static_actions(frame,shared_suppression_gate(frame))
        self.assertEqual(actions.tolist(),['alert','escalate'])

    def test_invalid_fallback_and_misaligned_gate_fail(self):
        frame=pd.DataFrame([signal(fallback_non_resource_action='escalate')]);gate=shared_suppression_gate(frame)
        with self.assertRaises(AssertionError):validate_signals(frame,gate)
        gate.index=[3]
        with self.assertRaises(ValueError):static_actions(frame,gate)

    def test_absolute_budget_does_not_depend_on_comparator_demand(self):
        saved=pd.DataFrame([{'scenario':s,'frontline_slots':2,'expert_slots':1}for s in SCENARIOS])
        policies={'clarity':pd.DataFrame({'pre_capacity_action':['escalate']*4+['alert']*2}),
                  'static':pd.DataFrame({'pre_capacity_action':['escalate']*3+['alert']*3})}
        limits=capacity_limits(saved,policies)
        self.assertEqual([(x['frontline_slots'],x['expert_slots'])for x in limits],[(2,1)]*3+[(3,4)])

    def test_alert_ties_exclude_protected_nonrationed_alerts(self):
        frame=pd.DataFrame({'pre_capacity_action':['alert','alert','alert'],
            'protected_alert':[True,False,False],'protected_review':[False,True,False],
            'highest_base_severity':[8,8,5],'delta_expert':[0,0,0],'delta_frontline':[1,1,1]})
        rows=priority_ties(frame,'test')
        self.assertTrue(all(x['candidates']==1 for x in rows if x['candidate_route']=='alert'))

    def test_return_zip_only_contains_explicit_aggregate_allowlist(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/'aggregate').mkdir();(path/'private').mkdir()
            (path/'aggregate/safe.json').write_text('{}');(path/'private/cases.csv').write_text('PRIVATE')
            z=write_return_zip(path,['safe.json'])
            with zipfile.ZipFile(z) as archive:self.assertEqual(set(archive.namelist()),{'safe.json','file_manifest.json'})

if __name__=='__main__':unittest.main()
