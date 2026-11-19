# TODO — implementácia

Pri úlohe doplňte `kto:`, po dokončení `[x]`. Úlohy s ⇄ sa dajú robiť súbežne.
Backprop píšeme ručne po vrstvách (`forward` + `backward`), operácie na celých maticiach (batch × príznaky).

## 1. Základ

- [ ] **1.1 ⇄ Matica / Tensor 2D** — kto: ___
  - `matmul` s poradím cyklov i-k-j, varianty s transponovaným A alebo B (bez kopírovania).
  - Pripočítanie biasu ku každému riadku, násobenie po prvkoch, `fill(0)`, súčet po stĺpcoch.
  - Výsledok zapisovať do predalokovaného výstupu (žiadne nové alokácie v tréningu).
  - Testy na malých maticiach s ručne spočítaným výsledkom.
- [ ] **1.2 ⇄ Načítanie dát** `data/*.csv` → matica `N × 784` (float, /255) + vektor labelov — kto: ___
  - Test: 60 000 trénovacích, 10 000 testovacích riadkov, prvý label sedí so súborom.
- [ ] **1.3 ⇄ Náhodné čísla** — `std::mt19937` so seedom 42, náhodná permutácia indexov — kto: ___
- [ ] **1.4 Build** — príkaz `g++ -std=c++17 -O3 -Iinclude src/*.cpp -o network` (do `run.sh`) — kto: ___

## 2. Vrstvy a loss

- [ ] **2.1 `Linear(in, out)`** — kto: ___
  - forward `Y = X·W + b`; backward `dW = Xᵀ·dY`, `db = súčet riadkov dY`, `dX = dY·Wᵀ`.
  - He inicializácia `N(0, √(2/in))`, bias 0.
- [ ] **2.2 `ReLU`** — forward `max(0, x)`, backward `dX = dY · [x > 0]` — kto: ___
- [ ] **2.3 Softmax + cross-entropy spolu** — kto: ___
  - Softmax stabilne (odčítať max riadku pred `exp`), loss `−log p[správna trieda]`.
  - Gradient `dZ = (softmax − one_hot) / batch_size`.
- [ ] **2.4 Sieť** — zoznam vrstiev, `forward` cez všetky, `backward` odzadu — kto: ___
- [ ] **2.5 Gradient check** — analytický vs. numerický gradient `(L(w+ε) − L(w−ε)) / 2ε` na malej sieti — kto: ___
- [ ] **2.6 XOR** — sieť 2 → 4–8 → 2 sa naučí všetky 4 prípady (overenie, že backprop funguje) — kto: ___

## 3. Tréning na Fashion-MNIST

- [ ] **3.1 Validácia** — z trénovacích dát odložiť 10 000 na validáciu (testovacie dáta len na finálnu predikciu) — kto: ___
- [ ] **3.2 Tréningová slučka** — minibatche (32–128), miešanie každú epochu, výpis loss, train/val accuracy, čas epochy — kto: ___
- [ ] **3.3 SGD** — prvá sieť 784 → 128 → 10, loss klesá — kto: ___
- [ ] **3.4 ⇄ Adam** (alebo momentum) — `m`, `v`, bias correction — kto: ___
- [ ] **3.5 ⇄ Regularizácia podľa potreby** — weight decay / dropout, zníženie lr po epochách — kto: ___
- [ ] **3.6 Ladenie** — architektúra, lr, batch, epochy; cieľ val accuracy ≥ 89 %, celý beh do ~6 min — kto: ___

## 4. Výstupy

- [ ] **4.1 Export** `train_predictions.csv` a `test_predictions.csv` do koreňa (jedno číslo 0–9 na riadok, poradie ako vstup) — kto: ___
- [ ] **4.2 `run.sh`** — kompilácia + `nice -n 19 ./network` — kto: ___
- [ ] **4.3 Kontrola** — `evaluator/evaluate.py` ukáže test accuracy ≥ 88 % — kto: ___
