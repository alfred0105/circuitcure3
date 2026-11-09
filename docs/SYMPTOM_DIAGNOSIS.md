# 증상 기반 PCB 진단 엔진

`circuitcare.symptom_diagnosis`에 `alfred0105/circuitcure2`의 증상 진단 엔진과 YAML 지식 베이스를 가져왔습니다. 이 엔진은 증상·관찰·측정값·선택 부품 후보를 받아 가능한 고장 모드를 근거와 함께 순위화합니다. 앱 프런트엔드는 저장소의 기존 `frontend/workbench-v2` 구현을 사용합니다.

## 로컬 실행 준비

```bash
python -m pip install -e .
ollama serve
ollama pull qwen3.5:9b
```

규칙 기반 엔진은 Ollama 없이 작동합니다. 이 환경에는 `qwen3.5:9b` (9.7B, `nvfp4`, 약 7.6GB)가 Ollama 모델 저장소에 내려받혀 있습니다. Ollama는 증상 문장 구조화 및 후보 설명을 돕는 선택 기능이며, 고장을 확정하거나 규칙 후보의 근거를 대체하지 않습니다.

## Python 사용

```python
from circuitcare.symptom_diagnosis import diagnose

result = diagnose(
    symptom_text="전원 표시등은 켜지고 버튼을 눌러도 버저가 울리지 않음",
    components=[{"component_id": "BZ1", "component_type": "buzzer"}],
)

for candidate in result["hypotheses"][:5]:
    print(candidate["title"], candidate["supporting_evidence"])
```

지식과 점수 규칙은 `src/circuitcare/symptom_diagnosis/knowledge/*.yaml`에 있습니다. 점수는 검증된 고장 확률이 아니며, 초기 규칙과 가중치는 실제 PCB 고장 사례로 검증되지 않았습니다. 측정 기반 순위 업데이트, UI 연결 및 실보드 정확도 평가는 별도 작업입니다.
