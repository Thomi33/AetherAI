import json, shutil, sys, time, requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "hf.co/peculiar-ragdoll/Sharp-MiniCPM5-2B-GGUF:Q6_K_XL"
PROMPT = sys.argv[2] if len(sys.argv) > 2 else "Explicame cómo funciona TCP en detalle."

rows = shutil.get_terminal_size().lines
out = sys.stdout
out.write(f"\033[2J\033[1;{rows-1}r\033[1;1H")   # limpia y reserva la última fila
out.flush()

r = requests.post("http://localhost:11434/api/chat", stream=True, json={
    "model": MODEL,
    "messages": [{"role": "user", "content": PROMPT}],
    "stream": True,
})

n, t0, last = 0, None, 0
d = {}
try:
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        if d.get("done"):
            break
        t0 = t0 or time.time()
        n += 1
        out.write(d["message"]["content"])
        now = time.time()
        if now - last > 0.2:                      # refresca stats 5 veces por segundo
            last = now
            tps = n / (now - t0) if now > t0 else 0
            out.write(f"\0337\033[{rows};1H\033[K[{n} tok | {tps:.1f} tok/s]\0338")
        out.flush()
finally:
    out.write("\033[r")                           # restaura la terminal
    out.write(f"\033[{rows};1H\033[K\n")

if d.get("eval_duration"):
    print(f"\nReal: {d['eval_count']/(d['eval_duration']/1e9):.1f} tok/s generación, "
          f"{d['prompt_eval_count']/(d['prompt_eval_duration']/1e9):.1f} tok/s prompt")
