# Аудио-блок: извлечение → анализ → препроцессинг — итоговый план (issue #25)

> Сводит в один план issue #25 и оба черновика из комментариев к нему: (A) «отказ от moviepy, режимы extract/direct» и (B) «предобработка аудио и сменные ASR-движки». Всё сверено с кодом на `main` @ `87c5ab2` (2026-09-28). Это **план**, а не описание текущей системы (текущее — `SYSTEM-SPEC.md`). После завершения этапа переносится в `docs/archive/`.
> Решения: D-008 … D-010 в `DECISIONS.md`.

## 1. Что и зачем

Сейчас звук из источника получается тремя разными путями: moviepy извлекает дорожку из видео в `.mp3`, Whisper декодирует файл сам через ffmpeg, диаризация ещё раз конвертирует его в WAV через `ensure_wav_16k_mono`, а длительность через moviepy меряется до трёх раз (`steps_ingestion.py:55`, `asr_engine.py:67`, `diarizer.py:70`). Оценки качества звука нет, доработки сигнала (нормализация, шумоподавление) нет.

Цель этапа — выделить подготовку звука в плоскую цепочку `PipelineStep` между загрузкой и ASR:

**проба → извлечение → подготовка (один декод) → анализ → препроцессинг → ASR → диаризация → …**

Оркестратор не меняется, каждый шаг сам решает «работать или пропустить себя».

## 2. Сверка с кодом: что в issue/черновиках устарело или уточнено

| Утверждение | Факт на `87c5ab2` | Вывод для плана |
|---|---|---|
| «Видео без звука — сейчас ничего не происходит» (issue §0) | `VideoIngestionStep` уже бросает `MediaDecodeError("...no audio track")`; CLI пишет `Skipping`, GUI показывает ошибку (`src/pipeline/steps_ingestion.py:35-38`) | Поведение не вводим, а **формализуем** (контракт `MediaInfo`) и добавляем данные пробы в лог |
| «FileLoader определяет тип и кладёт его в контекст» | `src/ingestion/file_loader.py` — только поиск файлов. Тип определяется по расширению в `src/service.py:36-37,176-186` | Контракт §0 — это поля `PipelineContext`, заполняемые шагом пробы |
| moviepy используется в 4 местах (черновик A) | Подтверждено: `video_extractor.py:6`, `asr_engine.py:15` (`_get_audio_duration_seconds`, его зовут `diarizer.py:70` и `steps_ingestion.py:16`); фолбэк `imageio_ffmpeg` в `system.py:83` | Как в черновике A |
| Метрики анализа пишутся в `document.metadata` | `TranscriptionStep` **заменяет** `context.document` целиком (`steps_transcription.py:24-33`) — всё записанное до ASR теряется | Метрики и провенанс живут в `PipelineContext` и переносятся в `document.metadata["audio"]` после ASR |
| В черновике B нет шага анализа, в issue нет `PreparedAudio`/пресетов | — | Совмещены ниже: анализ идёт на декодированном массиве, до препроцессинга |
| «Поддерживаемые форматы задаются в конфиге» | Три независимых списка: `general_config.yaml → extensions` (только поиск файлов в CLI), захардкоженные `_VIDEO_EXTS`/`_AUDIO_EXTS` в `src/service.py:36-37` (реальный шлюз, конфиг игнорирует), `file_types` в `app.py:218`. Воспроизведено: CLI находит `clip.wav`, сервис отвечает `MediaDecodeError: Unsupported file type '.wav'`. Whisper (ffmpeg) и диаризация (`pyannote` ресемплирует массив сам) `.wav` переваривают — отсекает именно список | Шлюз — проба файла, а не расширение (§4.4, C-006) |

## 3. Целевая цепочка шагов

```
service.transcribe_file — классификация по расширению (без изменений)
 1. Probe/Ingestion   probe_media() → context.media: MediaInfo; файл принят, если readable and has_audio (§4.4)
                      не читается / нет аудиопотока → MediaDecodeError (как сейчас) + поля MediaInfo в лог
 2. AudioExtraction   video + video_mode=extract → ffmpeg → audios/<stem>.wav ; direct → no-op ; audio → no-op
 3. AudioPreparation  один запуск ffmpeg → PreparedAudio.raw (float32, 16 кГц, моно)
 4. AudioAnalysis     метрики на raw (сигнал не меняется) → context.audio_report (+ рекомендация пресета)
                      «почти тишина» → SilentAudioError (подкласс MediaDecodeError) или warning — по конфигу
 5. AudioPreprocessing пресет ≠ none → повторный декод с цепочкой -af → PreparedAudio.asr / .diar
                      проверка: длина не изменилась (± 50 мс); провенанс → context.audio_provenance
 6. Transcription     Whisper на массиве; переносит audio_report + provenance в document.metadata["audio"]
 7. Diarization       pyannote на PreparedAudio.for_diarization (ensure_wav_16k_mono и .diarization_cache не нужны)
 8. Cleanup → Output  без изменений
 (нарезка на чанки — позже, строго после шага 5, по уже обработанному сигналу)
```

## 4. Контракты

### 4.1 `MediaInfo` (issue §0)

```python
@dataclass(frozen=True)
class MediaInfo:
    readable: bool            # ffmpeg открыл файл
    has_audio: bool           # есть хотя бы один аудиопоток
    duration: float | None    # сек; None, если ffmpeg не сообщил (напр. сырой .aac) — это НЕ ошибка
    audio_streams: int
    video_streams: int
    audio_codec: str | None
    sample_rate: int | None
    channel_layout: str | None  # как печатает ffmpeg: "mono", "stereo", "5.1"
```

Источник — вывод `ffmpeg -hide_banner -nostdin -i <file>` (stderr). Код возврата 1 бывает и у нормального, и у битого файла, поэтому `readable` определяется разбором stderr (`Input #0`), а не кодом возврата. ffprobe не используем: `imageio-ffmpeg` его не поставляет. `errors="replace"` при чтении — теги файла бывают не в UTF-8.

**Кроссплатформенность (Windows / macOS / Linux):** только `subprocess.run([...])` со списком аргументов (без shell), пути как `str(Path)`, бинарник — из `dependencies.ffmpeg_executable` после `ensure_ffmpeg_on_path` (системный ffmpeg или шим `imageio-ffmpeg`, `ffmpeg.exe` на Windows).

`video_streams` не считает обложки: ffmpeg показывает картинку в MP3/M4A как `Video: <mjpeg|png> ... (attached pic)` (проверено) — такой поток не делает файл видео.

### 4.2 Новые поля `PipelineContext`

| Поле | Тип | Кто пишет | Кто читает |
|---|---|---|---|
| `media` | `MediaInfo \| None` | шаг 1 | 2–7, лог |
| `audio_path` | `Path \| None` (есть) | шаг 2 | лог, бэкенды, которым нужен путь |
| `prepared_audio` | `PreparedAudio \| None` | 3, 5 | 4, 6, 7 |
| `audio_report` | `dict` | 4 | 5, 6 (→ metadata) |
| `audio_provenance` | `dict` | 2, 5 | 6 (→ metadata) |

### 4.3 `PreparedAudio`

```python
@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray        # float32, моно, [-1, 1]
    sample_rate: int = 16_000
    preset: str = "none"

@dataclass
class PreparedAudio:
    source_path: Path
    raw: AudioBuffer                 # без фильтров — вход анализа
    asr: AudioBuffer | None = None   # None ⇒ raw
    diar: AudioBuffer | None = None  # None ⇒ asr ⇒ raw

    @property
    def for_asr(self) -> AudioBuffer: return self.asr or self.raw
    @property
    def for_diarization(self) -> AudioBuffer: return self.diar or self.for_asr
```

**Декод без пресета повторяет путь Whisper бит-в-бит:** `-f s16le -ac 1 -ar 16000`, затем `/ 32768.0` — как `whisper.audio.load_audio`. Иначе (например, сразу `f32le`) Whisper получит другие отсчёты и транскрипты аудиофайлов могут измениться (INV-1).

Память: ~230 МБ на час звука на буфер; `asr`/`diar` создаются только при пресете ≠ `none`.

**Инвариант длины:** `len(asr) == len(diar) == len(raw)` (± 50 мс) — любой фильтр, меняющий длительность (`silenceremove`, `atempo`, `atrim`), запрещён, иначе разъедутся таймкоды, SRT/VTT и спикеры.

### 4.4 Входные форматы: принимаем всё, что ffmpeg декодирует как звук

ffmpeg читает сотни форматов (366 демультиплексоров в текущей сборке), включая картинки и субтитры, поэтому «принимать всё, что открывает ffmpeg» буквально нельзя. Правило:

- **Шлюз — `probe_media`:** файл принимается, если `readable and has_audio`. Иначе — `MediaDecodeError` с причиной (не читается / нет аудиопотока). Расширение больше не решает, принимать ли файл.
- **Видео или аудио** — по пробе: есть настоящий видеопоток (не `attached pic`) → видео (шаг извлечения), иначе → аудио. Расширение — только подсказка в логе.
- **Один список расширений** — в `general_config.yaml → extensions`, из него читают все: поиск файлов в папках CLI и фильтр загрузки в GUI (`app.py`). В сервисе захардкоженных списков нет. Список по умолчанию расширяется: аудио `.mp3 .m4a .aac .wav .flac .ogg .opus .wma .aiff`, видео `.mp4 .mov .avi .mkv .webm .m4v .wmv .flv .ts .mpg`. Он нужен только для удобства (что показывать/искать), а не как запрет.
- CLI по-прежнему ищет файлы по списку (а не «всё в папке»), чтобы не пробовать посторонние файлы. Новые расширения в списке меняют только то, какие файлы CLI находит, — транскрипты уже поддерживаемых файлов не меняются (INV-1).

## 5. Шаги

### 5.1 Извлечение (черновик A)
- Режимы `video_mode: extract | direct`, по умолчанию `extract` (текущее поведение: файл в `audios/` появляется).
- Формат извлечения по умолчанию меняется `.mp3` → `.wav` (16 кГц, моно, `pcm_s16le`) — ровно то, что нужно Whisper и pyannote. Расширение задаёт кодек (`.wav`, `.flac`, `.mp3`), остальное — `ValueError`.
- Запись во временный `.part`-файл + `replace`, чтобы прерванное извлечение не оставляло обрезанный файл.
- Приоритет настроек: CLI → конфиг → умолчание. Для аудиофайлов `video_mode` игнорируется.

### 5.2 Анализ (issue §2) — собственный расчёт на numpy, без новых зависимостей

Кадры по 30 мс, по ним RMS в dBFS.

| Метрика | Как считаем | Порог по умолчанию (`audio_analysis:` в `general_config.yaml`) | Что значит |
|---|---|---|---|
| `peak_dbfs` | max \|x\| | — | справочно |
| `rms_dbfs` | RMS всего файла | `< -35` → «тихо» | рекомендовать нормализацию |
| `clipping_ratio` | доля отсчётов \|x\| ≥ 0.999 | `> 0.001` → предупреждение | исправить нельзя, только сообщить |
| `silence_ratio` | доля кадров RMS < `-50` dBFS | `≥ 0.98` → «почти тишина» | warning или стоп (`on_silence: warn \| stop`, по умолчанию **`warn`** — без флагов поведение CLI не меняется, INV-1; стоп — по выбору пользователя) |
| `snr_estimate_db` | 95-й минус 10-й перцентиль энергии кадров | `< 15` → «шумно» | рекомендовать денойз (не применять автоматически) |

RMS кадра ограничивается снизу `-100` dBFS: цифровая тишина иначе даёт `log(0) = -inf` в перцентилях.

LUFS (EBU R128 через `pyloudnorm`) — не в первой итерации.

Результат — `audio_report = {"metrics": {...}, "flags": [...], "recommended_preset": "none|light|noisy"}`. Логируется всегда, независимо от того, применялась ли обработка.

### 5.3 Препроцессинг (issue §3, черновик B §7)

| Пресет | Фильтры ffmpeg | Риск |
|---|---|---|
| `none` | — | — |
| `light` | `highpass=f=80`, `dynaudnorm=f=250:g=15` | почти нет; может поднять шум в паузах |
| `noisy` | `highpass=f=80`, `afftdn=nf=-25`, `dynaudnorm=f=250:g=15` | артефакты на речи |
| `rnnoise` | `highpass=f=80`, `arnndn=m=<модель>`, `dynaudnorm=...` | нужен файл модели `.rnnn`; без него пресет недоступен |

- **Гибридная модель:** анализ всегда пишет рекомендацию; применение — только флагом `--audio-preset {none,auto,light,noisy,rnnoise}` / выпадающим списком в GUI. По умолчанию `none`. `auto` применяет рекомендацию, но не сильнее `light` — денойз только явно.
- Нормализация и шумоподавление — разные пресеты (разный риск-профиль).
- Порядок фильтров значим: срез низа → шумодав → нормализация.
- Препроцессинг всегда декодирует из **исходника** (`context.source_path`), а не из извлечённого 16-кГц `.wav`, и фильтры стоят **до** ресемплинга в 16 кГц. Модели `arnndn` обучены на 48 кГц, поэтому в пресете `rnnoise` сигнал сначала приводится к 48 кГц (`aresample=48000`), затем `arnndn`, затем 16 кГц.
- Проверка длины (±50 мс) для MP3-источников: задержка и паддинг кодировщика могут приближаться к допуску. С `.wav` по умолчанию (C-007) вопрос снимается; до этого в C-006 учитывать.
- Раздельные пресеты для ASR и диаризации (шумодав может ухудшать голосовые эмбеддинги); для диаризации — не сильнее `light`.
- Провенанс: пресет, итоговая строка `-af`, версия ffmpeg, время обработки → `metadata["audio"]["provenance"]`.
- Смена пресета по умолчанию — только по результатам WER-оценки (§8).

## 6. Конфигурация, CLI, GUI

```yaml
# configurations/general_config.yaml
output:
  extracted_audio_extension: ".wav"   # было ".mp3"
processing:
  video_mode: "extract"               # extract | direct
audio_analysis:
  silence_threshold_dbfs: -50
  silence_ratio_stop: 0.98
  on_silence: "warn"                  # warn | stop
  quiet_rms_dbfs: -35
  clipping_ratio_warn: 0.001
  low_snr_db: 15
audio_preprocessing:
  asr_preset: "none"                  # none | auto | light | noisy | rnnoise
  diarization_preset: "none"
  rnnoise_model: null
```

- CLI: `--video-mode {extract,direct}`, `--audio-preset {...}`. Без флагов — текущее поведение (INV-1).
- GUI: чекбокс «Сохранять извлечённую аудиодорожку», список «Улучшение звука». Правки `app.py` — минимальные (GUI вне фокуса блока).
- `transcribe_file(..., video_mode=None, audio_preset=None)` — аргументы, не записи в конфиг (INV-2).

## 7. Порядок работ: один чанк = один PR от `main`

В каждом чанке, помимо перечисленного, разрешены `docs/*` (BUILD-STATE, DECISIONS, AUDIT, SYSTEM-SPEC, CODEBASE-MAP — CLAUDE.md §4/§7) и `README.md` / `README.ru.md` (синхронно). Любой другой файл вне списка — блокер гейта.

| Чанк | Что | Ожидаемые файлы |
|---|---|---|
| **C-005** | `src/utils/ffmpeg_tools.py`: `probe_media`, `get_media_duration_seconds`, `extract_audio`, `decode_to_array` + тесты (медиа генерирует ffmpeg `lavfi` в фикстурах; в т.ч. MP3 с обложкой → не видео, `.wav`/`.flac`/`.ogg` → аудио, немое видео → `has_audio=False`). Существующий код не меняется | новый модуль, `tests/test_ffmpeg_tools.py`, `tests/conftest.py` |
| **C-006** | Уход с moviepy + шлюз форматов по пробе (§4.4): сервис принимает файл по `probe_media`, а не по `_VIDEO_EXTS`/`_AUDIO_EXTS`; видео/аудио — по пробе; единый список расширений из конфига для CLI и GUI (+ `.wav`, `.flac`, `.ogg`, …). `asr_engine`, `diarizer`, `steps_ingestion` на `ffmpeg_tools`; `MediaInfo` в контексте; удалить `video_extractor.py`; `requirements.txt`: −moviepy, +imageio-ffmpeg; повторить ручной гейт INV-9 `pip install --dry-run -r requirements.txt`. Формат пока `.mp3`, параметры кодирования максимально близки к moviepy (44.1 кГц, libmp3lame) | `src/transcription/{asr_engine,diarizer}.py`, `src/pipeline/steps_ingestion.py`, `src/models/document.py`, `src/ingestion/{__init__,video_extractor}.py`, `src/utils/system.py` (docstring), `src/service.py` (шлюз), `app.py` (`file_types` и превью из конфига), `configurations/general_config.yaml` (`extensions`), `requirements.txt`; тесты: `test_media_decode_error.py` (патчит `extract_audio_from_video`, `_probe_duration`), `test_service.py` (8 моков `_probe_duration`), `test_utils.py`, `test_ingestion.py`, `test_asr_engine.py` |
| **C-007** | `video_mode` + `.wav` по умолчанию | `src/service.py`, `src/utils/cli.py`, `main.py`, `app.py`, `configurations/general_config.yaml`, `src/pipeline/{__init__,steps_ingestion}.py`, `src/models/document.py` (`audio_provenance`), тесты |
| **C-008** | `PreparedAudio` + `AudioPreparationStep`: Whisper и pyannote на массиве; удалить `ensure_wav_16k_mono` и кэш `.diarization_cache` | `src/models/` (новый `audio.py`, `document.py`), `src/pipeline/{__init__,steps_transcription,steps_diarization}.py` + новый шаг, `src/transcription/{asr_engine,diarizer,audio_prep}.py`, `src/transcription/diarization_backends/{base,pyannote_backend}.py` (протокол принимает `AudioBuffer` вместо `Path`), `src/service.py`, тесты |
| **C-009** | `AudioAnalysisStep`: метрики, пороги, рекомендация, silence-gate, перенос в `metadata["audio"]` | новый шаг + модуль метрик, `src/models/document.py` (`audio_report`), `src/utils/{errors,__init__}.py` (`SilentAudioError`), `src/pipeline/{__init__,steps_transcription}.py`, `src/service.py`, конфиг, тесты |
| **C-010** | `AudioPreprocessingStep`: пресеты, `--audio-preset`, GUI, раздельные пресеты, провенанс | новый шаг, `src/pipeline/__init__.py`, `src/service.py`, `src/utils/cli.py`, `main.py`, `app.py`, конфиг, тесты |
| **C-011** | Оценка качества: `eval/run_eval.py` (WER/CER через `jiwer` — dev-зависимость), `eval/data/` в `.gitignore` | `eval/`, `.gitignore`, `requirements-dev.txt` |

Каждый чанк проходит полный цикл CLAUDE.md §4: re-ground → RED → GREEN → независимая проверка → адверсариальный гейт → доки → STOP.

## 8. Оценка качества (перед сменой умолчаний)

5–10 записей по 2–5 минут (чистая речь, созвон, шумное помещение, фон с музыкой, несколько спикеров) + ручные эталонные расшифровки. Матрица «модель × пресет» → WER/CER (нормализация: регистр, пунктуация, `ё→е`), время. Пресет становится умолчанием, только если ни на одной категории не ухудшает WER больше чем на ~1 п.п. и хотя бы на одной улучшает. Приватные записи не коммитятся.

## 9. Что НЕ трогаем

`src/pipeline/orchestrator.py`, `src/pipeline/steps.py`, `steps_cleanup.py`, `steps_output.py`, `src/output/formatter.py`, `src/history.py`, `src/utils/naming.py`, `src/processing/cleanup.py`, `src/transcription/alignment.py`, `src/transcription/diarization_config.py`, `src/utils/{device,logging_setup,config,progress}.py`, `configurations/params.yaml`.

Проверка в каждом PR: `git diff --stat main...HEAD` ⊆ «ожидаемые файлы» чанка из §7 + `docs/*` + README.

## 10. Инварианты

- **INV-1** — честная формулировка: **транскрипты аудиофайлов** без новых флагов должны остаться идентичными (декод без пресета повторяет путь Whisper бит-в-бит, §4.3). **Транскрипты видео могут измениться**: Whisper читает извлечённый файл, а меняется кодировщик (C-006: ffmpeg вместо moviepy, всё ещё MP3) и формат (C-007: `.wav` без потерь вместо MP3 с потерями). Это осознанное изменение (D-008), а не регрессия. Проверка в C-006, C-007, C-008: CLI без флагов на 2–3 эталонных файлах (аудио + видео) до/после; аудио — `diff` пустой; видео — дифф прикладывается к PR и оценивается человеком. `on_silence` по умолчанию `warn`, так что тихие файлы по-прежнему транскрибируются.
- **INV-2** — новые опции передаются аргументами, конфиг не пишется.
- **INV-3** — один резидентный Whisper; буферы `PreparedAudio` живут только в контексте одного задания.
- **INV-8** — автоопределение языка не ломается при переходе Whisper на массив (проверка в C-008).
- **INV-9** — в этапе 1 нет новых тяжёлых зависимостей (только ffmpeg, numpy, `imageio-ffmpeg`).
- **INV-10** — новые ошибки — подклассы `MediaDecodeError`, поэтому CLI пропускает файл, а пакет продолжается.

## 11. За рамками #25 (этап 2, отдельная issue)

- Сменный ASR-движок: `AsrBackend` + `OpenAIWhisperBackend`, опционально `WhisperXBackend` в отдельном `requirements-whisperx.txt` (жёсткий пин `torch~=2.8` конфликтует с текущим стеком — INV-9).
- Пословные таймкоды: `words` в `TranscriptSegment`, `WordAlignmentStep` (`src/transcription/word_alignment.py` — не путать с `alignment.py`, который назначает спикеров), пословное назначение спикеров.
- Нарезка на чанки (Phase 2.4, `merge_chunk_diarizations` — пока стаб).
- Demucs (отделение голоса от музыки).

## 12. Открытые вопросы (решаются в своём чанке)

1. Пороги §5.2 — стартовые; уточняются на реальных записях в C-009/C-011.
2. Для `on_silence: stop` — достаточно ли «почти тишины ≥ 98 %», или нужен и абсолютный порог по пику (иначе легитимно тихие записи будут остановлены)?
3. Источник модели RNNoise (`.rnnn`) и лицензия — до включения пресета `rnnoise` в C-010.
