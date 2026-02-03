# OpenClaw

Homepage: [openclaw.ai](https://openclaw.ai/)

I have installed OpenClaw on my Raspberry Pi 4 (RPI4).

[mcporter](https://github.com/steipete/mcporter/)

## Limitations

### npm

I had to switch to pnpm.

### uv

I had no luck installing `uv` on my RPI4. The `brew install uv` crashed when trying to install gcc.
Thus no way to run PlanExe locally on the RPI4.

## Clone PlanExe repo

```bash
euclid@raspberrypi:~ $ pwd
/home/euclid
euclid@raspberrypi:~ $ mkdir git
euclid@raspberrypi:~ $ cd git
euclid@raspberrypi:~/git $ git clone https://github.com/PlanExeOrg/PlanExe.git
Cloning into 'PlanExe'...
remote: Enumerating objects: 12618, done.
remote: Counting objects: 100% (529/529), done.
remote: Compressing objects: 100% (266/266), done.
remote: Total 12618 (delta 356), reused 356 (delta 257), pack-reused 12089 (from 3)
Receiving objects: 100% (12618/12618), 5.40 MiB | 9.65 MiB/s, done.
Resolving deltas: 100% (9189/9189), done.
euclid@raspberrypi:~/git $ cd PlanExe/mcp_local/
euclid@raspberrypi:~/git/PlanExe/mcp_local $ ls
AGENTS.md  planexe_mcp_local.py  README.md
euclid@raspberrypi:~/git $ 
```

/home/euclid/git/PlanExe/mcp_local/planexe_mcp_local.py


