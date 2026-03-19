# Repo Structure

Recommended layout for this project:

```text
guider-event-bot/
  docs/
    backend-architecture.md
    database-schema.sql
    api-contract.yaml
    repo-structure.md

  luma/
    task.md
    app/
    api/
    domain/
    db/
    jobs/
    tests/

  aibot_event management/
    task
    app/
    api/
    domain/
    db/
    realtime/
    tests/

  frontend/
    ...

  event-flow-app.html
  SYSTEM_DESIGN.md
```

## Service ownership

### `luma/`

This folder should contain:

- event publishing logic
- registration logic
- ticket issuance
- payment and receipt workflows
- notification jobs

### `aibot_event management/`

This folder should contain:

- scan and check-in workflows
- assistant/chat workflows
- feedback collection
- incident tracking
- live event state aggregation

