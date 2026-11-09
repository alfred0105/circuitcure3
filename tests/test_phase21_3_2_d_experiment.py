import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dslr_eval", ROOT / "scripts/evaluate_pcb_dslr_experiment.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_iou_xyxy_handles_exact_overlap_and_disjoint_boxes():
    assert EVAL.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert EVAL.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0


def test_one_class_match_counts_tp_fp_fn_and_board_recall():
    rows = [{"image_id": "pcb-a:1", "image_size_px": [640, 640], "board_id": "pcb-a",
             "ground_truth": [{"bbox_xyxy": [10, 10, 30, 30]}, {"bbox_xyxy": [100, 100, 120, 120]}],
             "predictions": [{"bbox_xyxy": [10, 10, 30, 30], "confidence": .9},
                             {"bbox_xyxy": [200, 200, 220, 220], "confidence": .8}] }]
    result = EVAL.match_at_iou(rows, .5, .25)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["precision"] == .5
    assert result["recall"] == .5
    assert result["f1"] == .5


def test_ap_and_small_object_recall_use_source_scale():
    rows = [{"image_id": "pcb-a:1", "image_size_px": [640, 640], "board_id": "pcb-a",
             "ground_truth": [{"bbox_xyxy": [10, 10, 20, 20]}],
             "predictions": [{"bbox_xyxy": [10, 10, 20, 20], "confidence": .8}]}]
    assert EVAL.average_precision(rows, .5, 640) == 1
    small = EVAL.match_at_iou(rows, .5, .25, small_only=True, imgsz=640)
    assert small["gt"] == 1 and small["recall"] == 1
