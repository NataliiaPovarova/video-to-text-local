import json
import logging
import socket
import urllib.error
import urllib.request

from .utils import ProcessingError


def cleanup_with_ollama(
    text: str,
    cleanup_model_name: str,
    cleanup_prompt: str,
    device: str,
    ollama_url: str,
    ollama_timeout_seconds: int,
    ollama_request_content_type: str,
    logger: logging.Logger,
) -> str:
    payload = {
        "model": cleanup_model_name,
        "prompt": f"{cleanup_prompt.strip()}\n\n{text.strip()}",
        "stream": False,
        "options": {"num_gpu": 1 if device == "cuda" else 0},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ollama_url,
        data=data,
        headers={"Content-Type": ollama_request_content_type},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=ollama_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except (TimeoutError, socket.timeout) as exc:
        raise ProcessingError(
            "Ollama request timed out while waiting for cleanup response. "
            f"Current timeout: {ollama_timeout_seconds}s. "
            "Try increasing 'ollama.timeout_seconds' in configurations/general_config.yaml "
            "or use a smaller/faster cleanup model."
        ) from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        if exc.code == 401 and body:
            try:
                parsed_body = json.loads(body)
                signin_url = parsed_body.get("signin_url")
                error_label = parsed_body.get("error")
                if signin_url:
                    raise ProcessingError(
                        "Ollama rejected the request as unauthorized. "
                        f"Sign in here to enable local access: {signin_url}"
                    ) from exc
                if error_label:
                    raise ProcessingError(f"Ollama unauthorized: {error_label}") from exc
            except json.JSONDecodeError:
                logger.debug("Could not parse Ollama 401 response body as JSON.")
        raise ProcessingError(f"Ollama HTTP error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise ProcessingError(
                "Ollama request timed out while waiting for cleanup response. "
                f"Current timeout: {ollama_timeout_seconds}s. "
                "Try increasing 'ollama.timeout_seconds' in configurations/general_config.yaml "
                "or use a smaller/faster cleanup model."
            ) from exc
        raise ProcessingError(
            f"Failed to reach Ollama at {ollama_url}. Ensure Ollama is running and the model is pulled."
        ) from exc

    parsed = json.loads(raw)
    cleaned = parsed.get("response", "")
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise ProcessingError("Ollama returned an empty cleanup response.")
    return cleaned.strip()
