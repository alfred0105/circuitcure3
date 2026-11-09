import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('phase_e_onnx',ROOT/'scripts/phase21_3_2_e_onnx_compare.py')
E=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(E)
EVSPEC=importlib.util.spec_from_file_location('phase_e_eval',ROOT/'scripts/evaluate_pcb_dslr_experiment.py')
EV=importlib.util.module_from_spec(EVSPEC);EVSPEC.loader.exec_module(EV)

def test_detection_parity_matches_same_class_boxes_and_scores():
    a=[{'class_id':0,'confidence':.8,'bbox_xyxy':[1,2,11,12]}]
    b=[{'class_id':0,'confidence':.79,'bbox_xyxy':[1.2,2,11.2,12]}]
    result=E.compare(a,b)
    assert result['matched_count']==1 and result['missing_or_extra_count']==0
    assert result['minimum_matched_iou']>.9

def test_detection_parity_reports_empty_mismatch_and_class_mismatch():
    a=[{'class_id':0,'confidence':.8,'bbox_xyxy':[1,2,11,12]}]
    assert E.compare(a,[])['missing_or_extra_count']==1
    wrong=[{'class_id':1,'confidence':.8,'bbox_xyxy':[1,2,11,12]}]
    assert E.compare(a,wrong)['matched_count']==0

def test_small_object_metrics_ignore_predictions_assigned_to_large_ground_truth():
    rows=[{'image_id':'x','image_size_px':[640,640],
        'ground_truth':[{'bbox_xyxy':[10,10,30,30]},{'bbox_xyxy':[100,100,220,220]}],
        'predictions':[{'bbox_xyxy':[100,100,220,220],'confidence':.9},
                       {'bbox_xyxy':[10,10,30,30],'confidence':.8}]}]
    result=EV.match_at_iou(rows,.5,confidence=.25,small_only=True,imgsz=640)
    assert (result['tp'],result['fp'],result['fn'])==(1,0,0)

def test_nms_free_decoder_parses_xyxy_scores_without_applying_nms():
    out=[[[5,6,20,24,.9,0],[10,10,30,31,.2,0]]]
    det=E.decode_nms_free_output(out,.25)
    assert det==[{'class_id':0,'confidence':.9,'bbox_xyxy':[5.,6.,20.,24.]}]
    assert E.decode_nms_free_output([], .25)==[]
    import pytest
    with pytest.raises(ValueError):E.decode_nms_free_output([[[1,2,3,4,5]]],.25)

def test_ic_benchmark_rejects_a_different_class_contract():
    class Model:
        names={0:'integrated_circuit',1:'resistor'}
    import pytest
    with pytest.raises(ValueError,match='expected class contract'):
        E.check_contract(Model())
