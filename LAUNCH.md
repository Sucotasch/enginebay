# Запуск Qwen3.6-27B как провайдер Hermes

## Быстрый старт (один клик)

### Windows CMD:
```bat
"D:\Arx\Software Downloads\Hermes copy\llm-inference-server\launch-hermes-llama.bat"
```

### Что делает:
1. Запускает llama-server (если ещё не запущен)
2. Ждёт загрузки модели (~20-30 сек)
3. Запускает Hermes с локальным провайдером

## Ручной запуск

### Шаг 1: Запустить сервер
```bat
"D:\Arx\Software Downloads\Hermes copy\llm-inference-server\start-llama.bat"
```

### Шаг 2: Запустить Hermes
```bash
hermes --provider local-qwen --model Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf
```

### Шаг 3: Остановить сервер
```bat
"D:\Arx\Software Downloads\Hermes copy\llm-inference-server\stop-llama.bat"
```

## Конфиг Hermes

Провайдер `local-qwen` добавлен в `~/.hermes/config.yaml`:

```yaml
providers:
  local-qwen:
    base_url: http://127.0.0.1:8080/v1
    api_key: not-needed
    models:
      - Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf
```

## Переключение провайдеров

### В Hermes chat:
```
/model local-qwen/Qwen3.6-27B.i1-IQ4_XS-attn_qkv-IQ4_XS.gguf
```

### Или через CLI:
```bash
hermes --provider local-qwen -q "Привет, как дела?"
```

## Производительность

| Метрика | Значение |
|---------|----------|
| Контекст | 96K (98304) |
| Decode | ~34 tok/s |
| Prefill | ~58 tok/s |
| VRAM | ~15.9 GB / 16 GB |
| Порт | 8080 |

## Файлы

```
llm-inference-server/
├── start-llama.bat          ← запуск сервера
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
- Проверьте что llama-server.exe существует:
  ```
  dir "C:\Users\sucot\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.20.1\llama-server.exe"
  ```
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
- Убедитесь что reasoning off: `--reasoning off`
