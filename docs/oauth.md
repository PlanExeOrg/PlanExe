---
title: OAuth setup
---

# OAuth setup for frontend_multi_user

This guide covers OAuth login for `frontend_multi_user` with:

- Google
- GitHub
- Microsoft
- Discord

It includes both production and localhost setups.

## How PlanExe builds callback URLs

PlanExe uses this pattern for all providers:

`{PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL}/auth/{provider}/callback`

Examples:

- `https://home.planexe.org/auth/google/callback`
- `https://home.planexe.org/auth/github/callback`
- `https://home.planexe.org/auth/microsoft/callback`
- `https://home.planexe.org/auth/discord/callback`
- `http://localhost:5001/auth/google/callback`
- `http://localhost:5001/auth/github/callback`
- `http://localhost:5001/auth/microsoft/callback`
- `http://localhost:5001/auth/discord/callback`

Important:

- `PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL` must have no trailing slash.
- If this env var is missing, PlanExe defaults to `http://localhost:5001`.

## Required environment variables

Set these on `frontend_multi_user`:

```bash
PLANEXE_FRONTEND_MULTIUSER_PUBLIC_URL="https://home.planexe.org"
PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY="insert-a-long-random-secret-for-sessions"
PLANEXE_AUTH_REQUIRED="true"

PLANEXE_OAUTH_GOOGLE_CLIENT_ID="insert-your-clientid"
PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET="insert-your-secret"

PLANEXE_OAUTH_GITHUB_CLIENT_ID="insert-your-clientid"
PLANEXE_OAUTH_GITHUB_CLIENT_SECRET="insert-your-secret"

PLANEXE_OAUTH_MICROSOFT_CLIENT_ID="insert-your-clientid"
PLANEXE_OAUTH_MICROSOFT_CLIENT_SECRET="insert-your-secret"

PLANEXE_OAUTH_DISCORD_CLIENT_ID="insert-your-clientid"
PLANEXE_OAUTH_DISCORD_CLIENT_SECRET="insert-your-secret"
```

Notes:

- Keep `PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY` stable across deploys, otherwise login sessions break.
- If `PLANEXE_AUTH_REQUIRED=true` and no OAuth provider is configured, startup fails by design.

## Credential storage

Track OAuth credentials in a password manager, for example 1Password.

- Store each provider's values (production and localhost) as separate entries.
- Also store the direct settings URL for each OAuth app/client in the same entry.
  Example: `https://discord.com/developers/applications/123456789012345/information`
- Never commit client secrets to git, docs, screenshots, or chat logs.

Provider naming is mostly the same:

- GitHub: `Client ID` and `Client Secret`
- Google: `Client ID` and `Client Secret` (OAuth 2.0 Client)
- Microsoft: `Application (client) ID` and `Client secret`
- Discord: `Client ID` and `Client Secret` (OAuth2 section)

## Production setup

Use your real public domain, for example `https://home.planexe.org`.

### Google - production

In Google OAuth client (`Web application`), add:

- `https://home.planexe.org/auth/google/callback`

Set:

- `PLANEXE_OAUTH_GOOGLE_CLIENT_ID`
- `PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET`

Verify what the app is using:

- Open `https://home.planexe.org/api/oauth-redirect-uri`
- Confirm `redirect_uri=` matches the URI in Google exactly.

### Google - localhost

In Google OAuth client (`Web application`), add:

- `http://localhost:5001/auth/google/callback`

Set:

- `PLANEXE_OAUTH_GOOGLE_CLIENT_ID`
- `PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET`

### GitHub - production

Create OAuth App at [github.com/settings/developers](https://github.com/settings/developers):

- Application name: `PlanExe`
- Homepage URL: `https://planexe.org/`
- Authorization callback URL: `https://home.planexe.org/auth/github/callback`
- Device Flow: off

Set credentials in:

- `PLANEXE_OAUTH_GITHUB_CLIENT_ID`
- `PLANEXE_OAUTH_GITHUB_CLIENT_SECRET`

### GitHub - localhost

Create OAuth App at [github.com/settings/developers](https://github.com/settings/developers):

- Application name: `PlanExe Localhost`
- Homepage URL: `http://localhost:5001/`
- Authorization callback URL: `http://localhost:5001/auth/github/callback`
- Device Flow: off

Set credentials in:

- `PLANEXE_OAUTH_GITHUB_CLIENT_ID`
- `PLANEXE_OAUTH_GITHUB_CLIENT_SECRET`

### Microsoft - production

In Azure Portal, create an app registration for PlanExe:
[portal.azure.com](https://portal.azure.com/)

Under **Authentication**, add redirect URI (Web):

- `https://home.planexe.org/auth/microsoft/callback`

Under **Certificates & secrets**, create a client secret.

Set credentials in:

- `PLANEXE_OAUTH_MICROSOFT_CLIENT_ID`
- `PLANEXE_OAUTH_MICROSOFT_CLIENT_SECRET`

### Microsoft - localhost

In the same Azure app registration, add redirect URI (Web):

- `http://localhost:5001/auth/microsoft/callback`

Set credentials in:

- `PLANEXE_OAUTH_MICROSOFT_CLIENT_ID`
- `PLANEXE_OAUTH_MICROSOFT_CLIENT_SECRET`

### Discord - production

Create an application in the Discord developer portal:
[discord.com/developers/applications](https://discord.com/developers/applications)

Name it `PlanExe`.

Open the OAuth2 page for your app (example):
`https://discord.com/developers/applications/1473810102153773206/oauth2`

Under OAuth2 settings, add this redirect:

- `https://home.planexe.org/auth/discord/callback`

Set credentials in:

- `PLANEXE_OAUTH_DISCORD_CLIENT_ID`
- `PLANEXE_OAUTH_DISCORD_CLIENT_SECRET`

Discord flow:

1. Open your app's OAuth2 page.
2. Copy `Client ID`.
3. Reset and copy `Client Secret`.
4. Under Redirects, add:
   `https://home.planexe.org/auth/discord/callback`

### Discord - localhost

Create an application in the Discord developer portal:
[discord.com/developers/applications](https://discord.com/developers/applications)

Name it `PlanExe Localhost`.

Open the OAuth2 page for your app (example):
`https://discord.com/developers/applications/1473810102153773206/oauth2`

Under OAuth2 settings, add this redirect:

- `http://localhost:5001/auth/discord/callback`

Set credentials in:

- `PLANEXE_OAUTH_DISCORD_CLIENT_ID`
- `PLANEXE_OAUTH_DISCORD_CLIENT_SECRET`

Discord flow:

1. Open your app's OAuth2 page.
2. Copy `Client ID`.
3. Reset and copy `Client Secret`.
4. Under Redirects, add:
   `http://localhost:5001/auth/discord/callback`

## Localhost setup (development)

Use `http://localhost:5001` as public URL.

## Troubleshooting

- `404` on `/login/<provider>`: provider env vars are missing.
  Required pairs: `PLANEXE_OAUTH_GOOGLE_CLIENT_ID`/`PLANEXE_OAUTH_GOOGLE_CLIENT_SECRET`, `PLANEXE_OAUTH_GITHUB_CLIENT_ID`/`PLANEXE_OAUTH_GITHUB_CLIENT_SECRET`, `PLANEXE_OAUTH_MICROSOFT_CLIENT_ID`/`PLANEXE_OAUTH_MICROSOFT_CLIENT_SECRET`, `PLANEXE_OAUTH_DISCORD_CLIENT_ID`/`PLANEXE_OAUTH_DISCORD_CLIENT_SECRET`.
- Redirect mismatch errors: callback URI in provider console does not exactly match PlanExe callback.
- Login does not persist after redirect: `PLANEXE_FRONTEND_MULTIUSER_SECRET_KEY` is missing or changed.
- Browser says insecure cookie on localhost over HTTP: expected in local dev; production should use HTTPS.
