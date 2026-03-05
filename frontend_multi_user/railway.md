# Railway Configuration for `frontend_multi_user`

See [PlanExe on Railway](https://docs.planexe.org/railway/) documentation for more details.

## Env vars

```
PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL="https://home.planexe.org"
PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD="insert-your-password"
PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME="insert-your-username"
PLANEXE_FRONTEND_MULTIUSER_PORT="5000"
PLANEXE_FRONTEND_MULTIUSER_DB_HOST="database_postgres"
PLANEXE_POSTGRES_PASSWORD="${{shared.PLANEXE_POSTGRES_PASSWORD}}"
PLANEXE_WORKER_PLAN_URL="http://${{worker_plan.RAILWAY_PRIVATE_DOMAIN}}:8000"
PLANEXE_AUTH_REQUIRED='true'
PLANEXE_OAUTH_GOOGLE_CLIENT_ID='insert-your-clientid'
PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET='insert-your-secret'
PLANEXE_OAUTH_GITHUB_CLIENT_ID='insert-your-clientid'
PLANEXE_OAUTH_GITHUB_CLIENT_SECRET='insert-your-secret'
PLANEXE_OAUTH_DISCORD_CLIENT_ID='insert-your-clientid'
PLANEXE_OAUTH_DISCORD_CLIENT_SECRET='insert-your-secret'
PLANEXE_GOOGLE_ANALYTICS="insert-your-secret"
PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY='insert-a-long-random-secret-for-sessions'
PLANEXE_STRIPE_SECRET_KEY='insert-your-secret'
PLANEXE_LLM_CONFIG_WHITELISTED_CLASSES="${{shared.PLANEXE_LLM_CONFIG_WHITELISTED_CLASSES}}"
PLANEXE_API_KEY_SECRET="${{shared.PLANEXE_API_KEY_SECRET}}"
```

## Session / admin login (production)

Set **PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY** to a long, random secret (e.g. `openssl rand -hex 32`). Flask uses it to sign the session cookie. If it is missing or changes between deploys, login (including admin) will not persist and you will see "Please log in to access this page" after signing in. When `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL` is HTTPS, the app sets the session cookie as Secure and SameSite=Lax so the browser sends it on redirects.

## OAuth

OAuth provider setup details are in `/docs/oauth.md`:

- Google (production + localhost)
- GitHub (production + localhost)
- Discord (production + localhost)


## Volume - None

The `frontend_multi_user` gets initialized via env vars, and doesn't write to disk, so it needs no volume.

## Domain

Configure a `Custom Domain` named `home.planexe.org`, that points to railway.
Incoming traffic on port 80 gets redirect to target port 5000.
