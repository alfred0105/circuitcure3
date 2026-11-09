import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('compare_inference',ROOT/'scripts/compare_component_experiment_inference.py')
COMPARE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(COMPARE)

def test_post_nms_detection_pairs_compare_scores_and_coordinates():
    box=[10.0,12.0,30.0,42.0]
    result=COMPARE.compare_rows([{'class_id':0,'confidence':.91,'bbox_xyxy':box}],
      [{'class_id':0,'confidence':.90,'bbox_xyxy':[10.2,12.1,30.2,42.1]}],.8)
    assert result['matched_count']==1 and result['unmatched_count']==0
    assert result['minimum_matched_iou']>.9
    assert result['maximum_confidence_abs_delta']<.02
    assert result['maximum_bbox_abs_delta_px']<1
