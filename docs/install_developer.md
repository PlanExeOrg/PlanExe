# Installing PlanExe for developers

I assume that you are a python developer.

You need several open terminals to do development on this project.

### Step 1 - Clone repo

```bash
git clone https://github.com/PlanExeOrg/PlanExe.git
```

### Step 2 - Prepare `.env` file

Create a `.env` file from the `.env.developer-example` file.

Update `OPENROUTER_API_KEY` with your open router api key.

### Step 3 - `open_dir_server`

In a new terminal: 
Follow the [`open_dir_server`](../open_dir_server/README.md) instructions.

### Step 4 - `worker_plan`

In a new terminal: 
Follow the [`worker_plan`](../worker_plan/README.md) instructions.

### Step 5 - `frontend_single_user`

In a new terminal: 
Follow the [`frontend_single_user`](../frontend_single_user/README.md) instructions.

### Step 6 - `database_postgres`

In a new terminal: 
Follow the [`database_postgres`](../database_postgres/README.md) instructions.

### Step 7 - `worker_plan_database`

In a new terminal: 
Follow the [`worker_plan_database`](../worker_plan_database/README.md) instructions.

### Step 8 - `frontend_multi_user`

In a new terminal: 
Follow the [`frontend_multi_user`](../frontend_multi_user/README.md) instructions.

### Step 9 - Tests

In a new terminal: 
Run the tests to ensure that the project works correctly.
```
PROMPT> python test.py
snip lots of output snip
Ran 117 tests in 0.059s

OK
```

### Now PlanExe have been installed.
