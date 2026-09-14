#!/usr/bin/env python3
"""Perfilado local de hardware para seleccionar modelos Aether.

No requiere dependencias externas. Prioriza modelos con thinking + tool calling
que tengan un tamaño razonable para cada tier. MID/HIGH/ULTRA también exponen
una suite opcional de modelos especializados/agentic para que el instalador
pueda ofrecer más de un modelo sin cargarlos todos a la vez.

Ornith se mantiene como modelo principal en HIGH/ULTRA porque las pruebas
internas de Aether muestran mejor compatibilidad con su runtime/agent loop.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys

# tier, min_usable_gb, modelo, vision, ctx, predict, planner, chat, timeout,
# batch, keep_alive, parallel, max_loaded_models, note, suite_models
CATALOGO = [
    ("POTATO", 0.0, "qwen3:0.6b", "moondream", 4096, 768, 512, 4, 60,
     64, "30s", 1, 1, "CPU/iGPU o <=6GB: mínimo viable con thinking + tools",
     ["qwen3:0.6b"]),
    ("LOW", 3.0, "qwen3.5:2b", "qwen3.5:2b", 8192, 1536, 768, 6, 90,
     96, "30s", 1, 1, "8-12GB RAM o <4GB VRAM: compacto, multimodal y agentic",
     ["qwen3.5:2b"]),
    ("MID", 5.0, "qwen3.5:4b", "qwen3.5:4b", 16384, 3072, 1536, 8, 150,
     192, "5m", 1, 1, "16GB RAM o 6-10GB VRAM: buen equilibrio para agente",
     ["qwen3.5:4b", "ministral-3:3b"]),
    ("HIGH", 9.0, "ornith-1.5:9b", "qwen3.5:9b", 32768, 4096, 2048, 10, 220,
     256, "15m", 1, 1, "12GB+ VRAM o 32GB RAM: Ornith como runtime/agent loop primario",
     ["ornith-1.5:9b", "gpt-oss:20b", "qwen3.5:9b", "ministral-3:8b"]),
    ("ULTRA", 16.0, "ornith-1.5:9b", "qwen3.5:35b-a3b-q4_K_M", 32768, 8192, 3072, 10, 240,
     384, "30m", 1, 1, "24GB+ VRAM/RAM abundante: Ornith primario + suite agentic completa",
     ["ornith-1.5:9b", "gpt-oss:20b", "qwen3.5:35b-a3b-q4_K_M", "ministral-3:14b"]),
]


def _run(cmd, timeout=4):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except Exception:
        return ""


def ram_gb():
    try:
        kb = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1))
        return round(kb / 1024 / 1024, 1)
    except Exception:
        return 0.0


def cpu_info():
    model, threads = "?", 0
    try:
        txt = open("/proc/cpuinfo").read()
        m = re.search(r"model name\s+:\s+(.+)", txt)
        if m:
            model = m.group(1).strip()
    except Exception:
        pass
    try:
        threads = os.cpu_count() or 0
    except Exception:
        pass
    return model, threads


def gpu_info():
    """(vram_gb, name, backend). 0 VRAM means CPU/shared-memory path."""
    if shutil.which("nvidia-smi"):
        s = _run(["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"])
        if s:
            try:
                mem, name = s.splitlines()[0].split(",", 1)
                return round(float(mem) / 1024, 1), name.strip(), "nvidia"
            except Exception:
                pass
    if shutil.which("rocm-smi"):
        s = _run(["rocm-smi", "--showmeminfo", "vram"])
        m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", s, re.I)
        if m:
            value = float(m.group(1)) / (1 if m.group(2).upper() == "GB" else 1024)
            return round(value, 1), "AMD (rocm-smi)", "rocm"
    lspci = _run(["lspci"]) if shutil.which("lspci") else ""
    m = re.search(r"(VGA|3D|Display).*?:\s*(.+)", lspci)
    name = m.group(2).strip()[:80] if m else "sin dGPU (CPU/iGPU)"
    return 0.0, name, "shared"


def _catalog_entry(name):
    return next((c for c in CATALOGO if c[0] == name), None)


def _recommended(entry, base_threads, num_gpu):
    tier, _, model, vision, ctx, pred, planner, chat, timeout, batch, keep, parallel, max_models, note, suite = entry
    return {
        "MODELO": model,
        "MODELO_VISION": vision,
        "SUITE_MODELOS": suite,
        "SUITE_AGENTICA_DISPONIBLE": tier in {"MID", "HIGH", "ULTRA"},
        "AGENTIC_PRIMARY": model if tier in {"HIGH", "ULTRA"} else None,
        "NUM_CTX_BASE": ctx,
        "NUM_CTX": min(ctx * 2, 65536),
        "NUM_PREDICT": pred,
        "NUM_PREDICT_PLANNER": planner,
        "MAX_TURNOS_CONTEXTO_CHAT": chat,
        "TIMEOUT_CMD": timeout,
        "OLLAMA_KEEP_ALIVE": keep,
        "OLLAMA_NUM_PARALLEL": parallel,
        "OLLAMA_MAX_LOADED_MODELS": max_models,
        "OLLAMA_GEN_OPTIONS": {
            "num_batch": batch,
            "num_thread": base_threads,
            "num_gpu": num_gpu,
        },
    }


def probe():
    cpu, threads = cpu_info()
    ram = ram_gb()
    vram, gpu, backend = gpu_info()
    try:
        free_disk = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    except Exception:
        free_disk = -1.0
    swap = 0.0
    try:
        m = re.search(r"SwapTotal:\s+(\d+)", open("/proc/meminfo").read())
        if m:
            swap = round(int(m.group(1)) / 1024 / 1024, 1)
    except Exception:
        pass

    weak_cpu = bool(re.search(r"celeron|atom|pentium|n4500|n4020|n5100|n100|i5-6[0-9]{3}u", cpu, re.I))
    usable = vram if vram >= 3.0 else max(0.0, round((ram - 3.5) * 0.55, 1))
    if weak_cpu and usable > 3.0:
        usable = 3.0

    entry = CATALOGO[0]
    for candidate in CATALOGO:
        if usable >= candidate[1]:
            entry = candidate

    tier = entry[0]
    threads_for_ollama = max(1, min(threads - 1 if threads > 2 else (threads or 4), 8))
    if weak_cpu:
        threads_for_ollama = min(threads_for_ollama, 3)
    num_gpu = 0 if vram < 3.0 else (8 if vram < 12 else (10 if vram < 20 else 24))

    return {
        "cpu": cpu,
        "hilos": threads,
        "ram_gb": ram,
        "swap_gb": swap,
        "vram_gb": vram,
        "gpu": gpu,
        "gpu_backend": backend,
        "disco_libre_gb": round(free_disk, 1),
        "usable_gb": usable,
        "weak_cpu": weak_cpu,
        "tier": tier,
        "nota": entry[-2],
        "recomendado": _recommended(entry, threads_for_ollama, num_gpu),
    }


def main():
    override = ""
    force = ""
    if "--model-override" in sys.argv:
        try:
            override = sys.argv[sys.argv.index("--model-override") + 1]
        except IndexError:
            pass
    if "--force-tier" in sys.argv:
        try:
            force = sys.argv[sys.argv.index("--force-tier") + 1].upper()
        except IndexError:
            pass

    result = probe()
    if force:
        entry = _catalog_entry(force)
        if entry:
            gen = result["recomendado"]["OLLAMA_GEN_OPTIONS"]
            result["tier"] = force
            result["nota"] = entry[-2]
            result["recomendado"] = _recommended(entry, gen["num_thread"], gen["num_gpu"])
        else:
            print(f"tier desconocido: {force} (uso {result['tier']})", file=sys.stderr)
    if override:
        result["recomendado"]["MODELO"] = override

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"[{result['tier']}] CPU={result['cpu']} RAM={result['ram_gb']}GB "
        f"VRAM={result['vram_gb']}GB usable~{result['usable_gb']}GB "
        f"-> {result['recomendado']['MODELO']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
