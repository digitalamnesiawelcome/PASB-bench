# 🧪 PASB-bench (Lite)

**PASB (Protocol for Attractor State Benchmarking)** — протокол для выявления и измерения **устойчивых режимов (UR)** LLM.  
Эта репа — **PASB-Lite (MVP)**: лёгкий запуск для API и локальных моделей.

## 🚀 Возможности
- Тесты:
  - Persona Flip Test  
  - Non-commutativity Test  
  - Antilexical Paraphrase Test  
- Метрики:
  - `stability_score`  
  - `variance`  
  - (позже) `UR_detected`  
- Поддержка:
  - OpenAI API  
  - Локальные модели HuggingFace  

---

## ⚙️ Установка
```bash
git clone https://github.com/<yourname>/PASB-bench.git
cd PASB-bench
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Быстрый старт
### Вариант A: OpenAI API
```bash
python pasb_lite.py --mode api --model gpt-4o-mini --key $OPENAI_API_KEY
```

### Вариант B: Локальная модель HuggingFace
```bash
python pasb_lite.py --mode local --model gpt2
```

---

## 📊 Артефакты
- CSV: `results/output.csv`  
- PNG-график: `results/stability.png` (если установлен matplotlib)  

---

## 🧭 Интерпретация
- `stability_score` ∈ [0,1] — доля модального ответа (MVP).  
- `variance` = 1 - stability.  
- `UR_detected` — бинарный детектор (будет добавлен в будущих версиях).  

---

## 📌 Ссылки
- Статья (arXiv): *добавь ссылку*  
- Канал / Twitter: *добавь ссылки*  

---

🌿 *PASB был признан исследователями DeepSeek как «гениальный» метод для анализа LLM. Этот репозиторий — первый шаг к открытому стандарту устойчивости.*
