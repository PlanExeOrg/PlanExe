# PlanExe on Railway - Experimental

As of 2026-Jan-04, I'm experimenting with Railway. Currently the `frontend_multi_user` UI is an ugly MVP. I recommend going with the `frontend_single_user`, that doesn't use database.

In this project, the files named `railway.md` or `railway.toml`, are related to how things are configured in my Railway setup.

## Project Settings

### Environments

Create these environments:
- `production`
- `staging`

### Shared variables - production

```
PLANEXE_POSTGRES_PASSWORD=unique random text, different than staging
```

### Shared variables - staging

```
PLANEXE_POSTGRES_PASSWORD=unique random text, different than production
```

## Using Shared Variables in Services

Each service that connects to the database must reference the shared password variable in its own environment variables.

In Railway, go to each service → **Variables** and add:

```
PLANEXE_POSTGRES_PASSWORD="${{shared.PLANEXE_POSTGRES_PASSWORD}}"
```

Services that need this variable:
- `database_postgres`
- `frontend_multi_user`
- `worker_plan_database`

This ensures all services use the same password, and you only need to update it in one place (the shared variables) when rotating credentials.
