# Phase 21.3.2-B — 클래스 매핑

배포 모델의 10-class 목록과 순서를 변경하지 않았습니다.

```text
resistor, capacitor, diode, led, integrated_circuit,
connector, switch_button, header, crystal, debug_test_point
```

## PCB-DSLR part 1

출처 annotation은 “all IC chips”를 회전 사각형으로 표시하고, 뒤의 선택 문자열은 IC 표면 text/OCR metadata입니다. 이 연구용 데이터는 **단일 `integrated_circuit` class**로만 변환했습니다. 실험 YAML 안에서 `integrated_circuit`의 ID는 0이고 배포 모델의 ID 4와 다릅니다. 실험 결과/weight를 기존 10-class ONNX에 대입하지 않습니다. 다른 부품 유형은 실험 목표 밖이므로 10-class training labels로 자동 전환하지 않습니다.

| Source data | Canonical/experiment label | 방식 |
|---|---|---|
| PCB-DSLR IC rotated box | `integrated_circuit` | Direct semantic class; rotated rectangle을 axis-aligned enclosing YOLO box로 변환 |
| PCB-DSLR trailing chip string | 없음 | OCR/text metadata이며 class가 아님 |
| PCB-DSLR 기타 PCB 부품 | 없음 | 별도 class 목표 밖; 이 데이터는 10-class 학습에 혼합하지 않음 |

## 보류된 후보 매핑

| 출처 표기 | 후보 해석 | 처리 |
|---|---|---|
| Cap1/2/3/4 | capacitor subtypes | 근거 검토 전 병합하지 않음 |
| Resistor | resistor | 후보 직접 대응; 별도 source 보류 |
| Resestor | 오타 가능성의 resistance class | 3 box뿐, 자동 수정/병합 금지 |
| MOSFET | MOSFET | transistor 또는 IC로 합치지 않음 |
| Mov | metal oxide varistor 추정 | diode로 합치지 않음 |
| Transformer | transformer | 별도 experimental class 후보 |
| WACV text/pads/pins/unknown/test 등 | non-component 또는 불확실 | canonical 10-class coverage 불완전 |
| FPIC SMD Designator / OCR / VTP | reference designator / text / connection | component class가 아님 |

Kaggle 후보는 source class 9종을 보존하는 [별도 실험 설정](../configs/component_training/pcb-component-detection-source-classes.yaml)을 사용합니다. 해당 데이터는 board ID·annotation completeness 때문에 `TRAINING_READY`가 아니며 PCB-DSLR IC-only 데이터와 합치지 않습니다.
