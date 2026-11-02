# ШАД · Семинар 5: AI-подсказки в разметке

## Где что лежит

- `seminar5_workbook.ipynb` — семинарный ноутбук с заданиями.
- `data/seminar5_challenge/cards.csv` — 100 карточек, gold и цены подсказок.
- `data/seminar5_challenge/baseline_project.json` — проект без подсказки.
- `data/seminar5_challenge/assisted_project.json` — проект с AI-подсказкой.
- `data/seminar5_challenge/instruction.html` — инструкция исполнителям.
- `src/workbook_support.py` — API-клиент Яндекс Заданий с журналом запросов.
- `src/home_experiment.py` — создание двух закрытых пулов и их отдельный запуск.
- `requirements.txt` — зависимости.

## Локальный запуск ноутбука

```bash
git clone https://github.com/sphsv/ysda-2026-seminar5-ai-hints.git
cd ysda-2026-seminar5-ai-hints
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
jupyter lab seminar5_workbook.ipynb
```

## Как повторить разметку через API

Запускайте код из корня репозитория. Сначала создайте два непересекающихся набора исполнителей и задайте им значения `0` и `1` одного скрытого навыка.

```python
from datetime import datetime, timezone, timedelta
from getpass import getpass
from pathlib import Path
import json
import pandas as pd

from src.workbook_support import TasksAPI
from src.home_experiment import stage_pair, request_launch

bundle = Path("data/seminar5_challenge")
cards = pd.read_csv(bundle / "cards.csv", dtype={"sample_id": str})
cards["hint"] = cards["hint_price"].map(lambda price: f"Цена товара — {price:,.2f} ₽")

baseline_rows = cards[["sample_id", "source_url"]].to_dict("records")
assisted_rows = cards[["sample_id", "source_url", "hint"]].to_dict("records")
baseline_project = json.loads((bundle / "baseline_project.json").read_text())
assisted_project = json.loads((bundle / "assisted_project.json").read_text())

cohort_skill_id = "YOUR_COHORT_SKILL_ID"
cohort_filters = {
    "baseline": {"and": [{"category": "skill", "key": cohort_skill_id,
                            "operator": "EQ", "value": "0"}]},
    "assisted": {"and": [{"category": "skill", "key": cohort_skill_id,
                            "operator": "EQ", "value": "1"}]},
}

budget_rub = 780.0
pool_settings = {
    "private_name": "seminar5_reproduction",
    "may_contain_adult_content": False,
    "will_expire": (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S"),
    "reward_per_assignment": 1.0,
    "assignment_max_duration_seconds": 600,
    "auto_accept_solutions": True,
    "mixer_config": {"real_tasks_count": 1, "golden_tasks_count": 0, "training_tasks_count": 0},
}

api = TasksAPI(
    getpass("OAuth-токен Яндекс Заданий: ").strip(),
    Path("data/home_runs/api_journal.json"),
    allow_write=True,
)

result = stage_pair(
    api, baseline_project, assisted_project, pool_settings,
    baseline_rows, assisted_rows, cohort_filters,
    overlap=3, budget_rub=budget_rub, commission=0.30,
)
print(result)
```

`stage_pair` создает проекты, задания и два **закрытых** пула. После проверки интерфейса, фильтров, оплаты и модерации каждый пул запускается отдельно:

```python
request_launch(api, result["pools"]["baseline"], budget_rub)
request_launch(api, result["pools"]["assisted"], budget_rub)
```

Запуск платный. Не коммитьте OAuth-токен и `data/home_runs/api_journal.json`.
