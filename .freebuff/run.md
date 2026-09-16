# Dashboard — como rodar (reprodução do ambiente de preview)

## Reproduzir artefatos não-commitados

- Copiar `.env` do checkout principal (`C:\Users\Yato\Desktop\kansai\.env`) para o mesmo
  caminho no worktree, se não existir. Adaptar `DASHBOARD_PORT`/`DASHBOARD_HOST` se a porta
  8799 estiver ocupada. As chaves `PROVIDER_<id>_API_KEY` também vivem no `.env`.
- Dependências: `pip install -r requirements.txt` no venv do projeto (`.venv` já existe no
  checkout principal).
- Nada mais: perfis (`profiles/*.yaml`, `*.memory.json`, `*.speakers.json`) ficam no
  repositório de trabalho, não no git.

## Rodar o servidor

Entrypoint correto: `python wallie.py --dashboard` (há um `serve()` em `dashboard/server.py`,
mas o driver oficial é o wallie). **Detached no Windows (PowerShell)** — usar `-PassThru` para
pegar o pid e `Start-Process` para sobreviver à sessão:

```powershell
$pid = (Start-Process -FilePath '.venv\Scripts\python.exe' `
  -ArgumentList 'wallie.py','--dashboard' `
  -WindowStyle Hidden `
  -RedirectStandardOutput '.freebuff\dash.log' `
  -RedirectStandardError '.freebuff\dash.err' `
  -PassThru).Id
```

- Env vars: `DASHBOARD_PORT=8799` (default do app; usar outra se ocupada) e `DASHBOARD_HOST=127.0.0.1`.
- stdout e stderr em arquivos DIFERENTES (PowerShell exige).
- Confirmar vida: `Get-Process -Id <pid>` e `curl http://127.0.0.1:8799/` → 200.
- O launcher do venv pode gerar um **processo filho** (launcher → python real): o pid que
  escuta a porta pode ser o filho. Confirmar com
  `netstat -ano | findstr :8799` e registrar o pid que LISTEN.

## Armadilha conhecida: servidores órfãos

Sessões anteriores deixaram `python -c "from dashboard.server import ..."` rodando com código
antigo preso à porta. Antes de subir, verificar/matar:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'dashboard|wallie' } |
  Select ProcessId, CommandLine
taskkill /F /PID <pid>
```

## Smoke test pós-start

- `GET /api/preflight` → lista de issues de configuração (vazio = tudo ok).
- `POST /api/start` → deve devolver `{"ok": true}` ou `{"detail": "<motivo real>"}` (nunca
  500 cru).
