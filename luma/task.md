Luma backend task

Purpose:
- own all pre-event workflows
- manage registration and approval
- issue tickets and QR payloads
- provide validation APIs to the AI bot

Suggested folder layout:
```text
luma/
  app/
  api/
  domain/
  db/
  jobs/
  tests/
```

Core responsibilities:
- event CRUD
- registration intake
- approval and waitlist logic
- ticket issuance
- QR token validation
- reminders and receipts
- organizer analytics

Primary data owned by Luma:
- organizations
- users
- events
- registration questions
- registrations
- registration answers
- tickets
- payments
- invoices
- notifications

External contract:
- validate QR tokens for the AI bot
- accept check-in writes from the AI bot
- expose event and attendee summary data
