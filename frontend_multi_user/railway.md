# Railway Configuration for `frontend_multi_user`

```
PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL="https://home.planexe.org"
PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD="insert-your-password"
PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME="insert-your-username"
PLANEXE_FRONTEND_MULTIUSER_PORT="5000"
PLANEXE_FRONTEND_MULTIUSER_DB_HOST="database_postgres"
PLANEXE_POSTGRES_PASSWORD="${{shared.PLANEXE_POSTGRES_PASSWORD}}"
PLANEXE_AUTH_REQUIRED='true'
PLANEXE_OAUTH_GOOGLE_CLIENT_ID='insert-your-clientid'
PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET='insert-your-secret'
PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY='insert-a-long-random-secret-for-sessions'
PLANEXE_STRIPE_SECRET_KEY='insert-your-secret'
```

## Session / admin login (production)

Set **PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY** to a long, random secret (e.g. `openssl rand -hex 32`). Flask uses it to sign the session cookie. If it is missing or changes between deploys, login (including admin) will not persist and you will see "Please log in to access this page" after signing in. When `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL` is HTTPS, the app sets the session cookie as Secure and SameSite=Lax so the browser sends it on redirects.

## OAuth (Google) in production

For "Sign in with Google" to work, two things must match exactly:

1. **Railway env:** Set `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL` to your public URL with no trailing slash, e.g. `https://home.planexe.org`. The app uses it to build the redirect URI: `{PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL}/auth/google/callback`.

2. **Google Cloud Console:** In your OAuth 2.0 Client (APIs & Services → Credentials → your OAuth client), under **Authorized redirect URIs**, add the **exact** URI your app uses. Open:
   ```
   https://home.planexe.org/api/oauth-redirect-uri
   ```
   You should see two lines: `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL=...` and `redirect_uri=...`. If the first shows `(not set)`, the env var is not reaching the app (check variable name, redeploy). Copy the **value** of `redirect_uri=` (the full URL) and add that exact string to **Authorized redirect URIs** in Google (one line, no trailing slash). Use the OAuth client type **Web application** and the client ID that matches `PLANEXE_OAUTH_GOOGLE_CLIENT_ID`. Save.

## Volume - None

The `frontend_multi_user` gets initialized via env vars, and doesn't write to disk, so it needs no volume.

## Domain

Configure a `Custom Domain` named `home.planexe.org`, that points to railway.
Incoming traffic on port 80 gets redirect to target port 5000.
