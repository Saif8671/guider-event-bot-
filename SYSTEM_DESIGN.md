# EventFlow System Design

This repository currently contains a single-file UI prototype in `event-flow-app.html`.
The prototype already demonstrates the end-to-end event lifecycle, so this document turns
that flow into a production system design and implementation pipeline.

## 1. Product Goal

EventFlow is an event operations platform that lets an organizer:

1. Create and publish an event
  add google maps intgration .
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




Event Registration Process.
Creating an Event
Collect Feedback from your Event Guests
Send a feedback email after your event to collect feedback on how the event went. Share positive feedback on social media.
Adding Hosts and Managers to Your Event
Easily manage events with your friends and team.
Event Referrals
Let guests invite their friends to your event and see which guests are sharing the event.
Check In Guests for In Person Events
Scan guests tickets to check them in for in person events. This helps you manage who joins your event and gives you attendance.
Zoom Integration automatically create Zoom Meetings or Webinars.
Event Insights
Understand how people are finding your event and registering.

Multi-Session / Recurring Events
Workarounds for multi-session events 

Updating Guest Information
Change a guest's status, tickets, name, or email

Event Guest List
Show or hide your event guest list on the event page



Updating Event Information
Keep your guests informed with changes to your event

Hybrid Events
Host events with both a physical and online location

Canceling an Event
You can cancel an event to delete the event and refund guests


Download Guest List as CSV
Export your event's guest list data for analysis, printing badges, or integration with external tools
Event Themes and Customization
Personalize your event with beautiful themes and visual effects
Cloning Events
Quickly duplicate an event with all its setting

Google Meet Integration
Host virtual events seamlessly with automatic Google Meet link generation

Integrating External Check-In Systems
Understanding QR code formats for third-party check-in applications

Waitlist
Manage overflow demand when your event reaches capacity

Managing Your Guest List
View, search, filter, and take actions on your event guests
Expanded Guest Table
Manage your guests efficiently with the expanded table view and actions menu

Event Cover Images
Guidelines and tips for creating great event cover images.

Mobile Wallet Passes
Add your event tickets to Google Wallet for quick access and offline availability

Contacting Event Hosts
How the Contact Host feature works for both guests and organizers

Getting Receipts for Your Tickets generate receipts or invoices 

Setting your Event Visibility

Sending or Scheduling Event Blasts
.
Event Terms
Learn how to create Terms question type, and optionally collect signature.

Setting Up Ticket Types
Create multiple ticket types for paid, free, sliding scale, require approval and more

Collect Registration Questions
We collect name and email for all guests, you can collect more information

Unlock Codes
Create special codes that unlock special tickets or discounts for your event

Calendars
Managing Events with Calendars
How calendars work and how to move events between them
Collaborating Calendars
Allow multiple calendars to manage the same event together.
Event Tags for Calendar Filtering
Create and assign tags to events so visitors can filter them on your calendar page
Sending Newsletters
Keep your community engaged with email newsletters


Deleting a Calendar
What to know before you say goodbye
Download Calendar People as CSV

Making Money
Creating a Paid Event
Sell tickets to your event.

Set up your Stripe account and collecting money for your events
Create Coupons for Paid Events
Give people free or discounted access to paid events.
Refunding a Guest
You can refund a guest's payment for an event.

Group Registration
Allow guests to register or purchase multiple tickets
Taxes and VAT
You can set a tax rate for all events under your calendar

Payment Methods
Learn how to accept credit card, Apple Pay, Google Pay and other payment methods.
Canceling an Event
You can cancel an event to delete the event and refund guests
Getting Receipts for Your Tickets
 generate receipts or invoices for your paid tickets.
Disputed Payments


Setting Up Ticket Types
Create multiple ticket types for paid, free, sliding scale, require approval and more
Payment + Require Approval
 supports collecting payment for tickets that require approval
Customize Invoices / Receipts
As an event host you can add your billing or tax info to receipts
Unlock Codes
Create special codes that unlock special tickets or discounts for your event
Understanding Payouts and Event Proceeds
Monitor your event proceeds and manage payouts.

Integrations
Integrate using our API or Zapier
Understanding Timezones 
 handles timezones for events and calendars


Managing Your Profile
Update your name, profile picture, bio, and social links
Contacting Event Hosts
How the Contact Host feature works for both guests and organizers



We've built a powerful, easy-to-use rich text editor that you can use for your newsletters, community library, and event descriptions.
Merging Luma Accounts
You can add multiple emails to your Luma account.
Printing Event Badges
 create and print badges for your event guests
Email Consent and Cold Emailing
We do not allow emailing people without their consent.

Create and manage events on your Android device
Download Guest List as CSV
Export your event's guest list data for analysis, printing badges, or integration with external tools
Chat 
Message other users, create group chats, and connect with event attendees
How to Promote Your Event and Grow Attendance
Practical strategies to get more people to your Luma events.
Event Themes and Customization
Personalize your event with beautiful themes and visual effects
Cloning Events


Expanded Guest Table
Manage your guests efficiently with the expanded table view and actions menu
.
Mobile Wallet Passes
Add your event tickets to Apple Wallet or Google Wallet for quick access and offline availability
Download Calendar People as CSV
Export your calendar's people list with subscriber data, membership information, and engagement metrics
Scheduling a Demo
You can try out Luma for free on our site
SMS / WhatsApp Messages
We send invites, reminders, and blasts with SMS / WhatsApp
Event Chat
Allow your event guests to chat with each other via event chat
Security
Single Sign-On (SSO) with Okta
Enterprise identity management and SSO integration
Two Factor Authentication
Secure your account with Two-Factor Authentication.
Active Devices
See where you're signed in and sign out of other devices.
Enterprise Security
Comprehensive overview of Luma's security practices and infrastructure for enterprise customers
Account Review and Appeal Process

Sign in with secure, passwordless passkeys on any device.
Security & Bug Bounty Program
Report security vulnerabilities and learn about our security practices
Troubleshooting
Updating Social Images
The social networks may store an old version of the image for your event. You can tell the social networks to use your new image.
Troubleshooting Google Calendar Invites
Luma should automatically add your events to your Google Calendar. You can see how that works and see how to troubleshoot it.
Email Delivery + Open Troubleshooting
Learn how email tracking works and how to troubleshoot any issues.
Site Accessibility and Troubleshooting
Solutions for when you can't access luma.com or experience connection issues
Location Accuracy and Event Discovery
Why you might be seeing events in the wrong location and how to fix it
Luma Plus
Luma Plus Overview
Premium features for power users and organizations
Understanding Import and Event Invite Limits
Different plans allow you to invite more people to your event but you can always have unlimited registrations.
Cancelling Luma Plus
You can cancel your Luma Plus at any time for any reason.
Facebook / Meta Tracking Pixel
Set up a Meta tracking pixel to track event registrations
Accepting Crypto Payments
Learn how to accept payments with Solana (SOL) and USDC for your events
Google Analytics Measurement ID
Set up a Google tracking pixel to track event registrations
Managing Luma Plus
You can update your payment method, turn off auto-renewal and more
Enterprise
Enterprise Overview
Advanced security, compliance, and support for organizations
Single Sign-On (SSO) with Okta
Enterprise identity management and SSO integration
Enterprise Security
Comprehensive overview of Luma's security practices and infrastructure for enterprise customers
Hardware Scanners for High-Volume Events
Integrate professional barcode scanning hardware for faster check-in at large-scale events
Transferring Tickets
How to transfer event tickets to other attendees