# EventFlow System Design

This repository currently contains a single-file UI prototype in `event-flow-app.html`.
The prototype already demonstrates the end-to-end event lifecycle, so this document turns
that flow into a production system design and implementation pipeline.

## 1. Product Goal

EventFlow is an event operations platform that lets an organizer:

1. Create and publish an event.
2. Collect registrations and optional approval requests.
3. Generate and deliver ticket QR codes.
4. Check guests in at the door.
5. Collect post-event feedback.
6. Review attendance and engagement analytics.

## 2. Prototype Flow Already Present

The current prototype covers these user journeys:

1. Event creation
2. Event page publishing
3. Guest registration
4. Guest management
5. QR ticket display
6. QR scan and check-in
7. Feedback submission

That means the prototype is already a valid UX map. The missing part is the production
architecture around persistence, auth, background jobs, payment delivery, and integrations.

## 3. Target System Architecture

### High-level layout

```text
Web Client
  -> API Gateway / Backend
    -> Auth Service
    -> Event Service
    -> Registration Service
    -> Ticket Service
    -> Check-in Service
    -> Feedback Service
    -> Notification Service
    -> Analytics Service
    -> Job Queue / Worker
    -> Database + Object Storage
    -> External Providers (email, SMS, payments)
```

### Recommended stack

This is the cleanest path for a real product:

1. Frontend: React or Next.js
2. Backend: FastAPI or NestJS
3. Database: PostgreSQL
4. Cache / queue: Redis
5. File storage: S3-compatible storage
6. QR generation: server-side library
7. Email delivery: SendGrid, Postmark, or SES
8. SMS / WhatsApp: optional, later phase
9. Payments: Stripe if paid tickets are required

## 4. Core Domains

### 4.1 Organizer

Represents the event host or team member managing the event.

Responsibilities:

1. Create events
2. Edit event details
3. Approve or reject registrations
4. Monitor attendance
5. Export guest lists
6. Review feedback

### 4.2 Event

Represents the event itself

Fields:

1. Name
2. Description
3. Date and time
4. End date and time
5. Location and location type
6. Cover media
7. Capacity
8. Ticket type
9. Approval mode
10. Speakers
11. Agenda / timeline
12. Custom registration questions

### 4.3 Registration

Represents a guest request or booking for an event.

States:

1. `pending`
2. `approved`
3. `declined`
4. `waitlisted`
5. `cancelled`

### 4.4 Ticket

Represents the unique identity guest uses for event entry.

Responsibilities:

1. Store ticket identity
2. Link to attendee
3. Support QR code issuance
4. Support check-in verification

### 4.5 Check-in

Represents a single entry attempt at the door.

Stored data:

1. Ticket ID
2. Timestamp
3. Device / staff member
4. Result
5. Duplicate status

### 4.6 Feedback

Represents post-event attendee response.

Collected fields:

1. Star rating
2. Free-text feedback
3. Return intent
4. Submitted at

## 5. End-to-End Pipeline

### Stage 1: Event Creation

Input:

1. Title
2. Date/time
3. Venue or virtual link
4. Description
5. Cover style
6. Capacity
7. Ticket type
8. Approval mode
9. Custom questions
10. Speakers
11. Timeline

Processing:

1. Validate required fields
2. Normalize dates and timezone
3. Save draft
4. Render preview
5. Publish event

Output:

1. Public event page
2. Shareable event URL
3. Registration form

### Stage 2: Registration Intake

Input:

1. Guest name
2. Guest email
3. Optional custom responses

Processing:

1. Validate email and required questions
2. Check capacity
3. Apply approval logic
4. Create registration record
5. Assign ticket status

Output:

1. Approved registration or pending approval
2. Confirmation screen
3. Ticket record

### Stage 3: Ticket Issuance

Processing:

1. Generate unique ticket token
2. Generate QR code payload
3. Store token hash in database
4. Deliver ticket by email
5. Provide in-app ticket view

Output:

1. Ticket email
2. QR code view
3. Wallet-ready asset later if needed

### Stage 4: Pre-Event Guest Management

Organizer actions:

1. Search guests
2. Filter by status
3. Approve pending registrations
4. Manually add guest
5. Export CSV
6. Re-send ticket

System behavior:

1. Update attendee counts
2. Recompute capacity availability
3. Log organizer actions

### Stage 5: Door Check-in

Input:

1. QR scan
2. Manual lookup fallback

Processing:

1. Parse ticket token
2. Verify signature / hash
3. Confirm registration status
4. Prevent duplicates
5. Record check-in event

Output:

1. Verified guest screen
2. Check-in success or duplicate warning
3. Attendance metrics update

### Stage 6: Post-Event Feedback

Input:

1. Rating
2. Return intent
3. Free-text comments

Processing:

1. Store response
2. Aggregate averages
3. Aggregate return intent

Output:

1. Feedback dashboard
2. Event quality metrics

## 6. Data Model

### `users`

1. `id`
2. `email`
3. `name`
4. `role`
5. `created_at`

### `events`

1. `id`
2. `owner_user_id`
3. `name`
4. `slug`
5. `description`
6. `start_at`
7. `end_at`
8. `location_type`
9. `location`
10. `capacity`
11. `ticket_type`
12. `price`
13. `approval_mode`
14. `status`
15. `cover_image_url`
16. `created_at`
17. `updated_at`

### `event_speakers`

1. `id`
2. `event_id`
3. `name`
4. `role`
5. `bio`
6. `avatar`
7. `sort_order`

### `event_timeline_items`

1. `id`
2. `event_id`
3. `time_label`
4. `title`
5. `description`
6. `icon`
7. `sort_order`

### `registration_questions`

1. `id`
2. `event_id`
3. `label`
4. `question_type`
5. `required`
6. `sort_order`

### `registrations`

1. `id`
2. `event_id`
3. `guest_name`
4. `guest_email`
5. `status`
6. `source`
7. `approved_by`
8. `created_at`
9. `updated_at`

### `tickets`

1. `id`
2. `registration_id`
3. `event_id`
4. `token`
5. `token_hash`
6. `qr_payload`
7. `issued_at`
8. `revoked_at`

### `checkins`

1. `id`
2. `ticket_id`
3. `event_id`
4. `checked_in_at`
5. `device_id`
6. `result`

### `feedback`

1. `id`
2. `event_id`
3. `registration_id`
4. `rating`
5. `attend_again`
6. `comments_like`
7. `comments_improve`
8. `submitted_at`

## 7. API Surface

### Auth

1. `POST /auth/login`
2. `POST /auth/logout`
3. `GET /auth/me`

### Events

1. `POST /events`
2. `GET /events/:id`
3. `PATCH /events/:id`
4. `POST /events/:id/publish`
5. `GET /events/:id/public`

### Registration

1. `POST /events/:id/register`
2. `GET /events/:id/registrations`
3. `PATCH /registrations/:id/approve`
4. `PATCH /registrations/:id/decline`

### Tickets

1. `GET /tickets/:token`
2. `POST /tickets/:token/resend`
3. `POST /tickets/:token/revoke`

### Check-in

1. `POST /checkins/verify`
2. `POST /checkins/manual`
3. `GET /events/:id/checkins`

### Feedback

1. `POST /events/:id/feedback`
2. `GET /events/:id/feedback/summary`

### Analytics

1. `GET /events/:id/dashboard`
2. `GET /events/:id/export.csv`

## 8. Background Jobs

Use a worker queue for anything slow or external.

Jobs:

1. Send registration confirmation email
2. Send QR code email
3. Send approval notification
4. Re-send ticket
5. Generate exports
6. Compute analytics snapshots
7. Send feedback reminders

## 9. Security and Reliability

### Security

1. Hash ticket tokens in storage.
2. Sign QR payloads.
3. Restrict organizer endpoints with role-based access.
4. Rate limit registration and check-in endpoints.
5. Validate all form input server-side.
6. Log audit actions for approvals and deletions.

### Reliability

1. Make check-in idempotent.
2. Use a transaction when issuing ticket and registration.
3. Queue outbound email so UI is not blocked.
4. Cache public event pages and counts if traffic grows.
5. Store exported files in object storage instead of memory.

## 10. Suggested Repo Structure

```text
eventflow/
  apps/
    web/
    api/
    worker/
  packages/
    ui/
    shared/
    validation/
  docs/
    SYSTEM_DESIGN.md
    PIPELINE.md
  infra/
    docker/
    terraform/
```

For this repository, the realistic first refactor is:

1. Convert the single-file prototype into a frontend app.
2. Add an API backend.
3. Add persistence.
4. Add worker jobs.
5. Add ticket delivery.

## 11. Build Phases

### Phase 1: Productize the Prototype

Deliverables:

1. Split HTML prototype into components.
2. Introduce routing.
3. Replace in-memory state with API data.
4. Preserve the current user flow.

### Phase 2: Backend Foundation

Deliverables:

1. Create event CRUD.
2. Create registration CRUD.
3. Create check-in verification.
4. Persist to Postgres.
5. Add auth.

### Phase 3: Ticketing and Delivery

Deliverables:

1. Generate unique tickets.
2. Email QR codes.
3. Support approval workflow.
4. Add resend and revoke flows.

### Phase 4: Operations

Deliverables:

1. Export CSV.
2. Analytics dashboard.
3. Feedback summaries.
4. Audit logging.

### Phase 5: Scale and Hardening

Deliverables:

1. Rate limits
2. Idempotency keys
3. Queued jobs
4. Monitoring and alerts
5. Multi-event organizer support

## 12. What I Would Build Next

If you want the next step implemented in code, the best order is:

1. Create a real app shell from the prototype.
2. Define backend models and APIs.
3. Wire the event create -> register -> ticket -> check-in pipeline.
4. Persist data.

That will turn the prototype into a working foundation instead of a static demo.
