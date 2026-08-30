# Запуск Qwen3.8-27B как провайдер Hermes

> Проект: **EngineBay** — репозиторий https://github.com/Sucotasch/enginebay
> Основная модель: `Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf` (ik_llama.cpp)

## Быстрый старт (один клик)

### Windows CMD (из папки проекта):
```bat
launch-hermes-llama.bat
```

### Что делает:
1. Запускает llama-server (если ещё не запущен)
2. Ждёт загрузки модели (~20-30 сек)
3. Запускает Hermes с локальным провайдером

## Ручной запуск

### Шаг 1: Запустить сервер
```bat
start-llama.bat
```
> Перед запуском укажите путь к модели:
> `set MODEL_GGUF=G:\Ai\Models\Qwen3.8-27B_qkv-IQ4_KS-MTP\Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf`

### Шаг 2: Запустить Hermes
```bash
hermes --provider local-llama --model Qwen3.8-27B.i1-IQ4_KT-attn_qkv-IQ4_KS-MTP.gguf
```

### Шаг 3: Остановить сервер
```bat
stop-llama.bat
```

## Конфиг Hermes

Провайдер `local-llama` автоматически регистрируется в `~/.hermes/config.yaml`
(см. AGENTS.md → Hermes + DeepSeek Harness integration):

```yaml
providers:
  local-llama:
    base_url: http://127.0.0.1:8080/v1
    api_key: not-needed
    models:
      - Local Model
```

## Переключение провайдеров

### В Hermes chat:
```
/model local-llama/Local Model
```

### Или через CLI:
```bash
hermes --provider local-llama -q "Привет, как дела?"
```

## Производительность (ik_llama.cpp, verified 2026-08)

| Метрика | Значение |
|---------|----------|
| Контекст | 96K (98304) |
| Decode | ~35.9 tok/s |
| Prefill | ~691 tok/s (313 tok) |
| VRAM | ~14.6 GB / 16 GB |
| Порт | 8080 |
| Prompt cache | 26.5s warmup → 0.86s повтор (~30x) |

## Файлы

```
enginebay/
├── start-llama.bat          ← запуск сервера (ik_llama, порт 8080)
├── start-beellama.bat       ← запуск BeeLlama (KVarN, порт 8080)
├── stop-llama.bat           ← остановка сервера
├── launch-hermes-llama.bat  ← запуск Hermes + сервер
├── configs/
│   └── inference.env        ← параметры сервера
├── scripts/
│   ├── start_llama_cpp.sh   ← лаунчер (Git Bash)
│   ├── start_llama_cpp.bat  ← лаунчер (Windows CMD)
│   ├── smoke_test.py        ← тест сервера
│   └── probe_hardware.py    ← детект железа
└── README.md
```

## Troubleshooting

### Сервер не запускается
- Проверьте что llama-server.exe существует (см. `start-llama.bat`, переменная `LLAMA_SERVER`)
- Проверьте что порт 8080 свободен:
  ```
  netstat -an | findstr 8080
  ```

### Hermes не видит провайдер
- Перезапустите Hermes
- Проверьте конфиг: `hermes config`
- Убедитесь что сервер работает: `curl http://127.0.0.1:8080/health`

### Медленная генерация
- Убедитесь что используется 96K контекст (не 128K!)
- Проверьте VRAM: `nvidia-smi`
- На Qwen3.8 (qwen35) нужен `--reasoning auto` + `--jinja` (не `off`)
- `-np 1` обязателен — авто-параллель даёт 4 слота и падение до ~3.8 tok/s
