<!-- Переключатель языка -->
[English](README.md) | **Русский**

# Расшифровка видео / аудио с локальным Whisper

Этот проект расшифровывает медиафайлы с помощью локально запущенной модели Whisper, после чего по желанию очищает транскрипт локальной моделью Ollama. У видеофайлов сначала извлекается аудиодорожка, а аудиофайлы (`.mp3`, `.m4a`) сразу отправляются на расшифровку. Результаты сохраняются в `transcripts/`, а логи выполнения — в `logs/`.

> **Памятка мейнтейнерам:** при правке этого файла, пожалуйста, синхронизируйте изменения с [`README.md`](README.md).

## Возможности

- CLI-флаг `--type` определяет, обрабатывать ли файлы из `videos/` или из `audios/`.
- CLI-флаг `--language` задаёт язык расшифровки Whisper (по умолчанию: `ru`).
- Автоматическое извлечение аудио из поддерживаемых видеоформатов (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
- Опциональный шаг очистки через локальный Ollama с использованием `cleanup_model` и `cleanup_prompt`.
- Опциональная диаризация спикеров через `pyannote.audio` (`--diarize`) с метками спикеров в выводе.
- Индикация прогресса во время расшифровки (оценка по длительности).
- Настройка моделей через `configurations/params.yaml`.
- Пути и параметры обработки через `configurations/general_config.yaml`.
- Структурированный код: корневой оркестратор (`main.py`) и доменные модули в `src/`.

## Структура проекта

```
.
├── audios/                     # Входные аудио или извлечённое аудио
├── videos/                     # Входные видео при запуске с --type video
├── transcripts/                # Результаты расшифровки + история GUI (создаётся автоматически)
├── logs/                       # Логи выполнения
├── configurations/
│   ├── general_config.yaml     # Пути, расширения, логирование и сервисы
│   ├── params.yaml             # Настройки моделей Whisper и очистки
│   ├── prompts.yaml            # Промпт для очистки через Ollama
│   └── diarization.yaml        # Настройки бэкенда диаризации спикеров
├── src/
│   ├── service.py              # Общий стык CLI+GUI: transcribe_file() (один файл, параметры на задание)
│   ├── history.py              # Индекс истории расшифровок для GUI
│   ├── ingestion/              # Поиск медиа и извлечение аудио из видео
│   ├── transcription/          # Whisper ASR + диаризация pyannote (diarizer, бэкенды, выравнивание, подготовка аудио)
│   ├── processing/             # Очистка через Ollama
│   ├── output/                 # Форматирование и запись транскриптов (txt, srt, vtt)
│   ├── pipeline/               # Оркестратор шагов и шаги конвейера (вкл. шаг диаризации)
│   ├── models/                 # Датаклассы (документ транскрипта, сегменты, контекст)
│   └── utils/                  # Разбор CLI, конфиг, устройство, логи, ошибки, ffmpeg, именование, прогресс
├── tests/                      # Набор тестов pytest
├── main.py                     # Точка входа CLI
├── app.py                      # Точка входа локального веб-интерфейса (Gradio)
├── transcribe                  # Скрипт-обёртка для удобства (macOS/Linux)
├── requirements.txt
├── requirements-gui.txt        # Дополнительные зависимости для веб-интерфейса
├── .env.example                # Шаблон для секретов (HF_TOKEN); скопируйте в .env
├── Dockerfile                  # Описание Docker-образа
└── .dockerignore
```

## Установка и настройка (локально)

### 1. Зависимости: `ffmpeg`

Скрипт использует `ffmpeg` для декодирования аудио. Убедитесь, что он доступен в PATH.

- **Windows (рекомендуется)**:
  В PowerShell от имени администратора:
  ```powershell
  choco install ffmpeg
  ```
- **macOS**:
  ```bash
  brew install ffmpeg
  ```

> **Замечание о диаризации:** `pyannote.audio >= 4.0` подтягивает
> `torchcodec` как транзитивную зависимость, а нативное расширение
> `torchcodec` на Windows ведёт себя нестабильно (несовместимость ABI
> разделяемых библиотек FFmpeg между минорными версиями, а также дрейф
> C++-ABI PyTorch). Чтобы избежать этого целого класса ошибок, бэкенд
> диаризации в проекте загружает аудио в память через
> `whisper.audio.load_audio` (вызывает ваш `ffmpeg.exe`) и передаёт
> `pyannote` словарь `{"waveform", "sample_rate"}`, полностью обходя
> `torchcodec`. Предупреждение `torchcodec is not installed correctly`
> от `pyannote` при старте является ожидаемым и безвредным —
> установленный `ffmpeg.exe` (даже static-сборка) полностью покрывает
> потребности проекта.
>
> На Windows тот же неудачный импорт может дополнительно вызывать
> модальное окно загрузчика DLL «точка входа в процедуру ... не найдена
> в библиотеке DLL `libtorchcodec_core8.dll`», которое иначе блокировало
> бы пакетные запуски до нажатия *OK*. В самом начале `main.py`
> вызывается `SetErrorMode(SEM_FAILCRITICALERRORS)` (до любого импорта,
> который может потянуть `torch` / `pyannote.audio` / `torchcodec`),
> чтобы подавить это окно для процесса. Текстовое предупреждение в
> stderr остаётся, поэтому проблема остаётся диагностируемой.

### 2. Зависимости: Ollama

Шаг очистки выполняется локально через Ollama. Установите его и загрузите модель для очистки:

```bash
ollama pull gpt-oss:latest
```

Если доступна CUDA, скрипт попросит Ollama использовать GPU для очистки.
Обратите внимание: некоторые модели, будучи локальными, требуют аутентификации.

### 3. Python-окружение

Создайте и активируйте виртуальное окружение:

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 4. Установка зависимостей

**Важно для пользователей NVIDIA GPU:**
Чтобы использовать ускорение CUDA, нужно установить версию PyTorch, совместимую с вашим GPU, **до** установки остальных зависимостей.

**Для RTX 50 Series (Blackwell) и новее:**
Требуется поддержка CUDA 12.8+.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Для более старых GPU (RTX 30/40):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Только CPU (или после установки PyTorch):**
Установите остальные зависимости:
```bash
pip install -r requirements.txt
```

## Конфигурация

Отредактируйте `configurations/params.yaml`, чтобы выбрать модель Whisper для расшифровки и модель для очистки:

```yaml
transcription_model: base     # tiny | base | small | medium | large-v3
cleanup_model: llama3.1:8b    # имя модели Ollama
```

Отредактируйте `configurations/prompts.yaml`, чтобы настроить инструкции для очистки:

```yaml
cleanup_prompt: |
  The text you receive is a transcription of a video / audio file. Please clean it up while preserving the meaningful information:
  - remove meaningless phrases like "oh, wait", "um", "got it", etc.;
  - split the text into paragraphs for easier reading;
  - if a part of the text contains a lot of meaningless artifacts, which might appear due to the poor quality of the sound, mark it as "[unclear]" and leave as is.
```

Отредактируйте `configurations/general_config.yaml`, чтобы управлять:
- путями ввода/вывода (`videos`, `audios`, `transcripts`, `logs`);
- поддерживаемыми расширениями медиа;
- именованием результатов (`.txt`, `_clean`, расширение извлечённого аудио);
- эндпоинтом и таймаутом Ollama;
- уровнем логирования, именем файла и форматом логов.

По умолчанию логи записываются в `logs/transcriber.log`.

## Использование

### Обработка видео
1. Поместите видеофайлы в `videos/`.
2. Запустите:
   ```bash
   python main.py --type video --language ru
   ```

### Обработка аудио
1. Поместите аудиофайлы в `audios/`.
2. Запустите:
   ```bash
   python main.py --type audio --language en
   ```

`--language` по умолчанию — `ru`, и его можно опускать при расшифровке русского.

### Опциональная очистка

Добавьте `--cleanup`, чтобы получить очищенные транскрипты через Ollama:

```bash
python main.py --type video --cleanup
```

### Опциональная диаризация спикеров

Диаризация использует `pyannote.audio` и требует токен доступа HuggingFace. Проект загружает секреты из локального файла `.env` при запуске (через `python-dotenv`).

**Первоначальная настройка:**

1. Скопируйте шаблон в реальный файл `.env` (не попадает в git):
   ```bash
   # Windows PowerShell
   Copy-Item .env.example .env

   # macOS / Linux
   cp .env.example .env
   ```
2. Откройте `.env` и пропишите свой токен:
   ```
   HF_TOKEN=hf_xxx_ваш_персональный_токен
   ```
   Создайте токен на https://huggingface.co/settings/tokens и примите пользовательские условия
   для каждой gated-модели, используемой `pyannote.audio` (для каждой нужно нажать «Agree and
   access» на отдельной странице HF):
   - `pyannote/segmentation-3.0`
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/speaker-diarization-community-1` *(требуется для `pyannote.audio >= 4.0`,
     поскольку оттуда подгружаются PLDA-веса даже при использовании модели
     `speaker-diarization-3.1`)*.

**Запуск:**

```bash
python main.py --type audio --diarize
```

Флаги диаризации:
- `--diarize` — включить диаризацию (переопределяет `diarization.enabled` в конфиге).
- `--no-diarize` — выключить её, даже если включена в конфиге (взаимоисключающе с `--diarize`).
- `--num-speakers N` — задать точное число спикеров (необязательно; иначе определяется автоматически).

> Каждый разработчик должен использовать свой персональный HF-токен. Файл `.env` находится в gitignore — никогда не коммитьте его. Переменная окружения `HF_TOKEN`, заданная в шелле или на уровне ОС, имеет приоритет над `.env`, что удобно для CI/CD.

**Поддерживаемые расширения:**
- Видео: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- Аудио: `.mp3`, `.m4a`

**Результаты:**
- Сырой транскрипт: `<name>.txt`
- Очищенный транскрипт: `<name>_clean.txt` (через Ollama, только с `--cleanup`)
- Логи: `logs/transcriber.log`

## Веб-интерфейс (локально)

Локальный однопользовательский веб-интерфейс (Gradio): загрузка файла, выбор
модели и языка, расшифровка с индикатором прогресса и просмотр истории.

```bash
pip install -r requirements-gui.txt   # один раз, после requirements.txt
python app.py                          # откроется http://127.0.0.1:7860
```

Интерфейс использует тот же движок, что и CLI; расшифровки и индекс истории
хранятся в `transcripts/`. Привязка только к `127.0.0.1` (без доступа из сети).

**Возможности интерфейса:**

- **Скачивание субтитров** — для каждого задания также создаются файлы `.srt` и
  `.vtt` с таймкодами, доступные для скачивания рядом с `.txt`.
- **Опциональная очистка** — отметьте *Clean up with Ollama*, чтобы обработать
  расшифровку локальной моделью Ollama. Если Ollama недоступна, задание всё равно
  завершается и сохраняет исходную расшифровку (предупреждение, а не ошибка).
- **Опциональная диаризация** — отметьте *Speaker diarization*, чтобы пометить
  спикеров. Требуется токен HuggingFace (`HF_TOKEN` в `.env`); без него флажок
  отключён с подсказкой. При желании можно задать точное число спикеров.
- **Пакетная загрузка** — загрузите один или несколько файлов сразу; они
  расшифровываются по очереди (очередь обрабатывает задания последовательно), и
  каждый попадает в историю.
- **Поиск по истории** — фильтрация таблицы истории в реальном времени по имени
  файла или дате.
- **Предпросмотр аудио** — для загруженных файлов `.mp3`/`.m4a` показывается
  встроенный плеер.

## Запуск через Docker

В проекте есть `Dockerfile`, который включает все зависимости. Учтите, что по умолчанию Docker-сборка использует CPU.

```bash
# Сборка
docker build -t whisper-transcriber .

# Запуск (пример для видео)
docker run --rm \
  -v "$(pwd)/videos:/app/videos" \
  -v "$(pwd)/audios:/app/audios" \
  -v "$(pwd)/transcripts:/app/transcripts" \
  -v "$(pwd)/configurations:/app/configurations:ro" \
  whisper-transcriber --type video
```
