# Dev Guide — EngineBay

**Что это:** рабочий конспект инженерных выводов по проекту, собранный в ходе
настройки launcher.py под несколько движков. Написан так, чтобы новый разработчик
(или агент после потери контекста) мог стартовать без повторного исследования.
Факты здесь проверены эмпирически на этой машине, если не сказано иное.

Дата последнего обновления: 2026-08 (vllm.cpp исследование, 96K pure конфиг).

---

## 0. Железо и окружение (эталон измерений)

| Компонент | Значение |
|---|---|
| GPU | RTX 4070 Ti SUPER, 16GB VRAM, sm_89 (Ada), arch `89` |
| CPU | i7-5820K, 6C/12T (сборки СЛОЖНЫЕ и ДОЛГИЕ — никогда не ставь маленькие таймауты) |
| CUDA | 13.1 (путь `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`) |
| MSVC | VS BuildTools 2022, MSVC 14.44 (`vcvars64.bat`) |
| CMake | 4.3.2 (в `D:\Works\Python`) |
| Ninja | `D:\Works\Python\Scripts\ninja.exe` (не используется, см. §7) |
| Git | 2.52 |
| Python | 3.x + PyQt6 (для launcher.py) |

Важные системные факты:
- **На PATH есть Strawberry Perl gcc** (`D:\works\Strawberry\c\bin\gcc.exe`).
  CMake с генератором Ninja без предварительно применённого vcvars может подхватить
  его вместо MSVC и зависнуть на "Detecting CXX compiler ABI info".
- CUDA runtime DLL (`cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`)
  НЕТ в bin CUDA Toolkit — копируются из папки beellama
  (`beellama.cpp/versions/preview-v0.4.4-cuda-13.1/`).
- `CUDA_PATH_V13_1` + `CUDA_PATH` должны быть установлены ДО вызова vcvars64,
  иначе MSBuild падает с "CUDA Toolkit directory '' does not exist".

---

## 1. Модели и их кванты (что чем читается)

Рабочая библиотека `G:\Ai\Models\` (проверено 2026-08):

| Файл | Архитектура | file_type | GGUF-тип | Читает движок |
|---|---|---|---|---|
| `Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf` | qwen35 | 145 | IQ4_KT/IQ4_KS (144/145) | **только ik_llama.cpp** |
| `qwen3.8-27b-IQ4_XS-pure.gguf` (13.54 GB) | qwen35 | 30 | IQ4_XS (23) | beellama, llama.cpp, vllm.cpp |
| `Qwen3.8-27B-i1-IQ4_XS-GGUF-Smaller.gguf` (12.61 GB) | qwen35 | — | IQ4_XS | beellama, llama.cpp |
| Gemma 4 12B/26B (`gemma-4-12B-it-heretic-Q8_0`, `gemma-4-26B-A4B-it-assistant.Q4_K_M`, `...Q6_K_L` и т.д.) | gemma4 | — | Q4_K_M/Q6_K_L/Q8_0 | все движки |
| Hearthfire/Magistry/Magnum/Precog 24B (`Q4_K_M`/`Q4_K_L`) | llama | — | Q4_K | все движки |
| mmproj-*.gguf (0.16-0.86 GB) | vision projector | — | F16/Q8_0 | — |

**Главное правило квантов:**
- **IQ4_KT/IQ4_KS — это треллис-кванты ik_llama (ggml type 144/145, выше 49).**
  Их НЕ читают beellama (останавливается на типе 49) и vllm.cpp (декантер знает
  максимум IQ4_XS=23, IQ1_XXXS=66, MXFP4=40, NVFP4=40).
- IQ4_XS (тип 23) и все стандартные k-quants — читаются всеми движками.

---

## 2. Параметры движков — совместимость и различия

Пресеты хранят ТОЛЬКО параметры (params/host/port) — **без пути к модели**.
Модель выбирается в GUI. Это осознанное решение пользователя.

Каждый движок имеет свои флаги для одних и тех же концепций — их НЕЛЬЗЯ
перемешивать:

| Концепция | llama.cpp (upstream) | beellama.cpp | ik_llama.cpp |
|---|---|---|---|
| KV cache | `--kv-unified --cache-type-k q4_0 --cache-type-v q4_0` | `--kv-unified --cache-type-k kvarn5 --cache-type-v kvarn4 --kv-tail-tokens 1024` | `--cache-type-k q4_0 --cache-type-v q4_0` (НЕТ `--kv-unified`) |
| Спекуляция | — | `--spec-draft-n-max 2 draft-mtp` | `--spec-type ngram-mod:... --spec-type mtp:...` (двухстадийная цепочка) |
| Reasoning | `--reasoning off` | `--reasoning off` | `--reasoning auto` (qwen35 ломается с `off`) |
| Jinja | `--jinja` | `--jinja` | `--jinja` — ОБЯЗАТЕЛЕН (tools) |
| Batch | `-b 2048 -ub 512` | `-b 2048 -ub 512` | `-b 1024 -ub 256` |
| Auto-offload | — | — | `--fit`, `--fit-margin N`, `--override-tensor "regex=CPU"` (ПРОБЕЛ, не `=`) |

**incompatible_params для ik_llama** (строки, наличие которых означает чужие
параметры): `("--kv-unified", "kvarn", "--kv-tail-tokens", "--spec-draft-n-max",
"draft-mtp")`.

**Механизм защиты в launcher.py:**
- `_load_saved_state`: если сохранённый конфиг содержит чужие флаги → сброс на
  `default_params` текущего движка + лог "Params from a different engine detected".
- `_on_engine_changed`: при смене движка в комбобоксе params заменяются на
  дефолты нового движка.
- `_on_preset_selected` + `_find_incompatible`: при выборе пресета, несовместимого
  с текущим движком, — диалог "Engine mismatch" с выбором: загрузить дефолты
  движка или применить пресет как есть (advanced).

---

## 3. VRAM-математика для Qwen3.8-27B (16GB карта)

Измерено на ik_llama (IQ4_KT/KS MTP модель, 14.0 GB файл):

- **Pure mode** (без `--spec-type`): CUDA0-буфер **константа 12780.93 MiB**
  при любом контексте (KV выделяется отдельно). 128K влезает впритык (15572 MiB).
- **MTP mode**: добавляет голову blk.64 (~216 MiB) + MTP-KV (~72 MiB) →
  влезает только до 48K. На 56K/64K — переполнение, ~3.9 t/s.
- **KV self (q4_0)**: 64K = 1224 MiB, 48K = 918, 32K = 576.
- **CPU-буфер**: константа 1288.28 MiB (output layer 1220.59 MiB f16) —
  НЕ влияет на скорость ни при каком контексте.

**Правила для этой модели на 16GB:**
- ≥64K контекста = только pure mode. MTP на ≥64K физически невозможен.
- q4_0 KV — минимально допустимое качество (пользователь: ниже q4 — "галюциногенная
  каша"; q2_0 на этой модели вообще крашит сервер, 0xC0000409).
- FFN-offload на CPU (i7-5820K) убивает decode: 8 слоёв → 5.68 t/s, 4 → 10.13.
  Никакие "редкие" слои (через 8, 20, 40) не спасают — любой offload бьёт по скорости.
- `--fit` (авто-offload) — тупик: при марже 0 сгружает 2806 MiB на CPU.
- `-np 1` ОБЯЗАТЕЛЕН для qwen35: авто-parallel создаёт 4 слота × 96K KV → VRAM
  не влезает, ~3.8 t/s.

**Итоговый рабочий конфиг (96K pure, ~35.9 t/s):**
```
-c 98304 -np 1 -ngl 99 -b 1024 -ub 256 --cache-type-k q4_0 --cache-type-v q4_0
-t 5 -tb 6 --flash-attn on --jinja --reasoning auto --temp 1.0 --min-p 0.0
--top-p 0.95 --top-k 20 --presence-penalty 0.0 --repeat-penalty 1.0
--no-mmproj-offload
```
Проверено end-to-end: tools-запрос → `reasoning_content` + `tool_calls`,
35.87 t/s. Hermes съедает ~20K контекста → ~76K полезного.

---

## 4. Измеренные скорости (референс)

| Конфиг | t/s | Комментарий |
|---|---|---|
| Pure 80K | 35.19 | влезает (14654 MiB) |
| Pure 96K | 35.65 | **рабочий** (14960 MiB) |
| Pure 128K | 32.13 | впритык (15572 MiB) |
| MTP 48K | 47.35 | максимум для MTP, контекст мал |
| MTP 56K/64K | ~3.9 | переполнение KV/MTP-головы |
| MTP head на CPU 64K | 28.17 | медленнее pure → бессмысленно |
| FFN 8 слоёв на CPU | 5.68 | offload = тупик на этом CPU |
| FFN 4 слоя на CPU | 10.13 | offload = тупик |

Вывод пользователя: **MTP существует ради скорости; если он не ускоряет при
нужном контексте — компромисс бессмыслен.** → 96K pure это выбор.

---

## 5. Сборка ik_llama.cpp из исходников

См. `build-ikllama.bat` (самодокументированный). Ключевые моменты:

1. `ik_llama.cpp` НЕ имеет prebuilt Windows-бинарей (tag t0002, 0 assets).
   В папке `versions/<commit>/` кладутся: `llama-server.exe`, `ggml.dll`,
   `llama.dll`, `mtmd.dll` + CUDA runtime DLL из beellama.
2. CMake: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_LLAMAFILE=OFF`.
3. **Параллелизм: `cmake --build ... --config Release --parallel N` (дефолт 6).**
   ⚠️ Если сборку запускает агент через файловую песочницу — MSBuild молча
   деградирует до 1 процесса (песочница блокирует named pipes worker-нод).
   Вне песочницы (двойной клик / свой терминал) /m:N параллелит нормально.
4. Ninja-генератор НЕ использовать: cmake+Ninja зависает на ABI-детекте под
   песочницей (подхватывает Strawberry gcc или блокируется pipes).
5. Сборка занимает 30-90 мин на i7-5820K (одним процессом — часы). Не прерывать.

---

## 6. vllm.cpp (исследовано 2026-08, клон в `vllm_research/`)

**Статус:** Windows-релиз ещё не опубликован (спека `windows-binary-release.md`
status=ACTIVE, publication=pending). CI собирает `windows-x86_64-msvc-cpu` и
`-vulkan` на `windows-2022` через `scripts/build-windows-release.ps1`. Нативный
Windows CUDA-build сырой: открытые issues #1475, #1478, #1479 (GCC-прагмы,
POSIX unistd, CUDA 13.2 + Windows).

**Ключевые факты:**
- Архитектуры: qwen35 (dense+MoE+MTP), gemma2/3/4, llama, qwen3_moe, deepseek_v2/v4,
  и др. — наша библиотека покрыта.
- Кванты: стандартные до IQ4_XS (23), IQ1_XXXS (66), MXFP4/NVFP4 (39/40).
  **IQ4_KT/IQ4_KS (144/145) НЕ поддерживаются.**
- Движок: PagedAttention + continuous batching (порт vLLM). Выигрыш при МНОГИХ
  параллельных стримах; одиночный стрим — не быстрее llama.cpp.
- Сборка требует CUTLASS 4.5.0 fetch (~200 MB) для FA2.

**Вывод:** для текущей задачи (один Hermes-агент + IQ4_KT/KS модель) vllm.cpp
не даёт преимуществ. Следить за Windows-релизом, пробовать на IQ4_XS-pure
для сравнения paged KV — если появится.

---

## 6b. syv-ai/qwen38-27b-rtx3090 (референс, клон в `qwen3090_research/`)

vLLM-стек (Python + safetensors W4A16 AutoRound, НЕ GGUF), Qwen3.8-27B на одной
RTX 3090 24GB, 150K контекст. **Референс-ориентир скорости, не инструмент для
нас** (нужно 24GB, формат не читается нашими движками).

**Измеренные там числа (RTX 3090, 250W):**
- single-user MTP: **121 tok/s** (greedy 120); DFlash2: 127-133 tok/s
- копирование/цитирование контекста: **381 tok/s** (lookup/drafting из промпта)
- batch (64 conc): ~1035 tok/s агрегат
- повторный запрос к тому же документу: **0.56s vs 22.4s** (PREFIX_CACHE)
- KVarN 4/2-bit KV cache: **268K контекст** на 24GB

**Что это дало НАМ (эмпирически проверено на ik_llama):**
- **ik_llama имеет prompt cache в RAM по умолчанию** (`--cache-ram 8192`,
  `--cache-ram-similarity 0.50`, `--cache-ram-n-min`). Это аналог PREFIX_CACHE
  vLLM. Для qwen35 работает (`llama_model_supports_partial_kv_reuse` → true).
  Отключается только для openPangu+MTP.
- **Замер на нашей машине** (96K pure, IQ4_KT/KS, префикс ~20K токенов):
  warmup 26.46s → повторные запросы **0.86s / 0.84s** (f_keep: 1.00,
  cache 30965 токенов). Это ~30x на повторных запросах — критично для
  Hermes-агента, который шлёт один и тот же системный префикс (~20K).
- **Hermes-вывод:** пока контекст не переполняется, повторные запросы почти
  бесплатны. `--cache-ram 0` отключает (не делать).

**Что НЕ переносится к нам (для памяти):**
- vLLM-specific патчи (DFlash2 backport #52816, Marlin int8, int4-KV-per-token-head).
- KVarN-порт на vLLM 0.27.1 (у нас KVarN уже есть в beellama: kvarn5/kvarn4).
- 150K/268K контекст недостижим на 16GB с моделью 14GB.

---

## 7. Ловушки и уроки (проверено на практике)

1. **BOM в JSON ломает launcher.** `load_presets`/конфиг читаются через
   `read_text(encoding="utf-8")` — UTF-8 BOM (который добавляет
   `Set-Content -Encoding UTF8` в PowerShell) роняет json.load
   ("Unexpected UTF-8 BOM"). Переписывать файлы без BOM
   (python `json.dump(..., encoding="utf-8")` или `-Encoding utf8NoBOM`).
2. **`--override-tensor` в ik_llama — через ПРОБЕЛ**, не `--override-tensor=...`
   (форма с `=` → сервер выходит с 1 и печатает help).
3. **`--jinja` обязателен** для tools-запросов Hermes: без него сервер отвечает
   "tools param requires --jinja flag".
4. **Скрытое окно сервера:** Popen требует
   `creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`, иначе при запуске
   из GUI выскакивает консоль.
5. **Dry-run:** `llama-server -dr --dry-run ...` пропускает загрузку тензоров —
   удобно проверять layout/память без 2-минутной загрузки 14GB.
6. **Select-String по огромным файлам** (148k строк лога) — таймаут; работать
   точечно по диапазонам строк.
7. **Пресеты не хранят модель** — только params/host/port. Это требование
   пользователя: пресет применяет параметры к тому, что выбрано в GUI.
8. **Порты:** 8080 = Qwen3.6/3.8, 8888 = Gemma/Agentic. Не смешивать.
9. **Клиентская интеграция (Hermes + DSH):** launcher автосинхронизирует
   `~/.hermes/config.yaml` (provider `local-llama`, base_url
   http://127.0.0.1:PORT/v1, model "Local Model") И `~/.dsh/settings.yaml`
   (provider `local_llama` под `llm-pi-ai.providers`, поле `baseURL`).
   Механизм DSH-синка: `_dsh_upsert_local_llama(content, host, port)` —
   чистая модульная функция; обновляет порт в существующем блоке и вставляет
   новый блок, если его нет. Трогается ТОЛЬКО блок `local_llama` — остальные
   провайдеры и top-level секции не изменяются (проверено diff'ом: 181 строка
   без изменений). settings.yaml пользователя читается через PyYAML-валидацию
   после записи; при ручных правках хранить бэкап `settings.yaml.pre-*`.

---

## 8. Проверка изменений (минимум перед коммитом)

```bash
python -m py_compile launcher.py
python -c "import json; json.load(open('launcher_presets.json', encoding='utf-8'))"
python -c "import json; json.load(open('launcher_config.json', encoding='utf-8'))"
```

End-to-end (запуск сервера + tools-запрос): см. историю сессии — поднять сервер
с текущим конфигом, POST /v1/chat/completions с tools, проверить `tool_calls` в
ответе и строки "eval time" в логе.
