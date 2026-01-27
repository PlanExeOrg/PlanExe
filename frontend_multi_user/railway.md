# Railway Configuration for `frontend_multi_user`

```
PLANEXE_FRONTEND_MULTIUSER_ADMIN_PASSWORD="insert-your-password"
PLANEXE_FRONTEND_MULTIUSER_ADMIN_USERNAME="insert-your-username"
PLANEXE_FRONTEND_MULTIUSER_PORT="5000"
PLANEXE_FRONTEND_MULTIUSER_DB_HOST="database_postgres"
PLANEXE_POSTGRES_PASSWORD="${{shared.PLANEXE_POSTGRES_PASSWORD}}"
```

## Volume - None

The `frontend_multi_user` gets initialized via env vars, and doesn't write to disk, so it needs no volume.

## Domain

Configure a `Custom Domain` named `home.planexe.org`, that points to railway.
Incoming trafic on port 80 gets redirect to target port 5000.
