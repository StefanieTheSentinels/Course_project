# ab_blackbox

Симулятор A/B-тестов и поиск оптимальной конфигурации кнопки методами чёрного ящика
(DIRECT, Differential Evolution, Bayesian Optimisation).

Курсовой проект, НИУ ВШЭ, 2-й курс.

---

## О чём проект

Задача — найти параметры кнопки (цвет фона и текста, размер, шрифт, отступы,
качество текста, позиция на странице), при которых CTR максимален.

В реальности каждое измерение CTR — это запуск A/B-теста с тысячами пользователей,
поэтому функция дорогая и шумная. Это типичный сценарий **black-box optimisation**.

Пайплайн:

1. **Ground-truth формула** — `p_click(params)` собрана из эмпирических исследований
   (WCAG, Baymard, Chartbeat, NN/g, Infolinks, VWO).
2. **Синтетический датасет** — генерируется случайными конфигами + бинарными кликами
   через формулу.
3. **ML-модель** учится предсказывать `p(click)` по фичам — она и есть «чёрный ящик».
4. **Оптимизация** — три метода ищут максимум CTR, делая вызовы к симулятору.
5. **Анализ результатов** — A/B-тест с z-критерием, SRM-чек, бутстрап CI.

---

## Структура

```
ab_blackbox_project/
├── ab_blackbox/                  # библиотека
│   ├── generating_formula.py     # ground-truth p_click
│   ├── model.py                  # ButtonModel, FullSyntheticModel, TrainedMLModel
│   ├── simulator.py              # BlackBox — симулятор A/B
│   ├── experiment.py             # run_ab_test
│   ├── analysis.py               # z-test, SRM, bootstrap
│   ├── datasets.py               # генератор синтетических данных
│   └── training.py               # фичи + обучение sklearn
├── tests/
│   └── test_ab_blackbox.py       # pytest
├── generate_dataset.py           # CLI: датасет
├── train_models.py               # CLI: обучение
├── optimize.py                   # CLI: сравнение оптимизаторов
├── sensitivity_analysis.py       # CLI: чувствительность к весам формулы
├── demo.py                       # end-to-end демо
├── _run_all.sh                   # пайплайн целиком
└── requirements.txt
```

---

## Быстрый старт

```bash
pip install -r requirements.txt

# полный пайплайн (5 прогонов с разными seed):
chmod +x _run_all.sh
./_run_all.sh
```

Или по шагам:

```bash
# 1. сгенерировать датасет
python3 generate_dataset.py --n 120000 --seed 42

# 2. обучить модель
python3 train_models.py

# 3. сравнить оптимизаторы (DIRECT, DE, BO)
python3 optimize.py --seed 42 --plot-out convergence.gif

# 4. проверить устойчивость к весам формулы
python3 sensitivity_analysis.py

# 5. тесты
pytest tests/ -v
```

---

## Результаты

### Пример CTR от оптимизаторов

| Метод | Calls | Best CTR | Runtime |
|---|---|---|---|
| Bayesian Optimisation (GP) | 200 | 0.978 | ~190 s |
| Differential Evolution | 1344 | 0.968 | ~0.3 s |
| DIRECT | 203 | 0.781 | ~0.05 s |

BO находит лучшую конфигурацию за меньшее число вызовов чёрного ящика, ценой
бо́льшего runtime (фит GP на каждом шаге).

### Конвергенция

`optimize.py` строит анимированный GIF: слева — кривые best-so-far, справа —
живая визуализация найденной кнопки для каждого метода.

![convergence](convergence_1.gif)

### Чувствительность к весам формулы

`sensitivity_analysis.py` варьирует `beta_size`, `beta_text`, `lambda` по сетке
и проверяет, остаётся ли оптимум стабильным. Результат сохраняется в
`sensitivity_results.csv`.

---

## Ключевые компоненты

### `BlackBox` — симулятор

```python
from ab_blackbox import BlackBox, FullSyntheticModel

box = BlackBox(model=FullSyntheticModel(), n_users=10_000, seed=42)
result = box({
    "rgb_bg": (255, 255, 255), "rgb_text": (0, 0, 0),
    "btn_w": 200, "btn_h": 60, "font_size": 18,
    "text_quality": 0.95, "whitespace_ratio": 0.35,
    "scroll_to_button": 0.15, "hour": 13, "device": "desktop",
})
print(result.ctr)  # → ~0.30
```

### Запуск A/B-теста

```python
from ab_blackbox import run_ab_test, analyze, print_report

exp = run_ab_test(box, params_A, params_B)
report = analyze(exp, alpha=0.05, run_bootstrap=True)
print_report(report)
```

Вывод:
```
SRM Check: [OK] No SRM  (p=0.7821)
Primary metric (CTR):
  A=0.1240  B=0.1530  delta=+0.0290  (+23.4% lift)
  z=4.812  p=0.0000  [SIG]
DECISION: SHIP
```

---

## Ground-truth формула

```
p(click) = p_attractive × penalty × visibility
```

- **p_attractive** = `sigmoid(β₀ + β_text·tq + β_time·(t−1) + β_ws·ws)`
- **penalty** = произведение независимых штрафов (контраст, размер, гармония
  цвета, баланс отступов, overflow)
- **visibility** = `f_position_decay(scroll)` с экспоненциальным затуханием

Источники: WCAG 2.1, Baymard 2024, Apple HIG, Google Material, Chartbeat 2014,
NN/g 2018, Itten 1961, Ou & Luo 2006, Infolinks, VWO/Open Mile.

---

## Аргументы CLI

`optimize.py`:
```
--budget INT         бюджет вызовов BlackBox (по умолчанию 200)
--n-users INT        пользователей на оценку CTR (10000)
--noise FLOAT        Gaussian noise на p_click (0.01)
--device STR         "mobile" | "desktop"
--seed INT           seed
--plot-out PATH      путь к GIF
```

`generate_dataset.py`:
```
--n INT              размер датасета (50000)
--seed INT
--noise FLOAT        Gaussian noise на p_click (0.02)
--out PATH
```

`train_models.py`:
```
--data PATH          входной CSV (synthetic_dataset.csv)
--cv INT             CV-фолды (5)
--out PATH           путь к pickle модели (best_model.pkl)
```

---

## Зависимости

```
numpy, scipy, pandas, scikit-learn, scikit-optimize, matplotlib, sympy, pytest
```

Точные версии — в `requirements.txt`.

---
---

# ab_blackbox (English)

A/B-test simulator and black-box optimisation of button configurations
(DIRECT, Differential Evolution, Bayesian Optimisation).

Course project, HSE University, 2nd year.

## What this is

Goal: find button parameters (background/text colour, size, font, padding, text
quality, vertical position) that maximise CTR.

In practice every CTR measurement is a full A/B test with thousands of users —
the objective is expensive and noisy. This is a textbook **black-box
optimisation** problem.

Pipeline:

1. **Ground-truth formula** — `p_click(params)` built from empirical UX research
   (WCAG, Baymard, Chartbeat, NN/g, Infolinks, VWO).
2. **Synthetic dataset** — random configs + Bernoulli clicks via the formula.
3. **ML model** is trained to predict `p(click)` from features — this is the
   "black box".
4. **Optimisation** — three methods search for max CTR by querying the simulator.
5. **A/B analysis** — two-proportion z-test, SRM check, bootstrap CI.

## Quick start

```bash
pip install -r requirements.txt

chmod +x _run_all.sh
./_run_all.sh
```

Step by step:

```bash
python3 generate_dataset.py --n 120000 --seed 42
python3 train_models.py
python3 optimize.py --seed 42 --plot-out convergence.gif
python3 sensitivity_analysis.py
pytest tests/ -v
```

## Results

| Method | Calls | Best CTR | Runtime |
|---|---|---|---|
| Bayesian Optimisation (GP) | 200 | 0.978 | ~190 s |
| Differential Evolution | 1344 | 0.968 | ~0.3 s |
| DIRECT | 203 | 0.781 | ~0.05 s |

BO finds the best configuration with the fewest black-box calls but spends
more time fitting the GP at each step.

## Components

```python
from ab_blackbox import BlackBox, FullSyntheticModel, run_ab_test, analyze

box = BlackBox(model=FullSyntheticModel(), n_users=10_000, seed=42)
result = box(params)
print(result.ctr)

exp = run_ab_test(box, params_A, params_B)
analyze(exp, alpha=0.05, run_bootstrap=True)
```

## Ground-truth formula

```
p(click) = p_attractive × penalty × visibility
```

- `p_attractive = sigmoid(β₀ + β_text·tq + β_time·(t−1) + β_ws·ws)`
- `penalty` = product of independent penalties (contrast, size, colour harmony,
  margin balance, overflow), each in (0, 1]
- `visibility = f_position_decay(scroll)` — sweet-spot at 15% then exponential
  decay

Sources: WCAG 2.1, Baymard 2024, Apple HIG, Google Material, Chartbeat 2014,
NN/g 2018, Itten 1961, Ou & Luo 2006, Infolinks, VWO/Open Mile.

## Dependencies

`numpy, scipy, pandas, scikit-learn, scikit-optimize, matplotlib, sympy, pytest`
— pinned in `requirements.txt`.