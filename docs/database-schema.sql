-- Luma schema

CREATE TABLE organizations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  role TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE events (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  title TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  description TEXT,
  status TEXT NOT NULL,
  start_at TIMESTAMP NOT NULL,
  end_at TIMESTAMP NOT NULL,
  timezone TEXT NOT NULL,
  location_type TEXT NOT NULL,
  location_name TEXT,
  location_address TEXT,
  virtual_url TEXT,
  capacity INTEGER,
  approval_mode TEXT NOT NULL,
  ticket_type TEXT NOT NULL,
  cover_image_url TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE registration_questions (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  label TEXT NOT NULL,
  field_key TEXT NOT NULL,
  question_type TEXT NOT NULL,
  required BOOLEAN NOT NULL DEFAULT FALSE,
  sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE registrations (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  guest_name TEXT NOT NULL,
  guest_email TEXT NOT NULL,
  guest_phone TEXT,
  status TEXT NOT NULL,
  source TEXT NOT NULL,
  approved_by TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE registration_answers (
  id TEXT PRIMARY KEY,
  registration_id TEXT NOT NULL REFERENCES registrations(id),
  question_id TEXT NOT NULL REFERENCES registration_questions(id),
  answer_text TEXT,
  answer_json TEXT
);

CREATE TABLE tickets (
  id TEXT PRIMARY KEY,
  registration_id TEXT NOT NULL UNIQUE REFERENCES registrations(id),
  event_id TEXT NOT NULL REFERENCES events(id),
  token_hash TEXT NOT NULL UNIQUE,
  token_prefix TEXT NOT NULL,
  qr_payload TEXT NOT NULL,
  issued_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at TIMESTAMP
);

CREATE TABLE payments (
  id TEXT PRIMARY KEY,
  registration_id TEXT NOT NULL REFERENCES registrations(id),
  event_id TEXT NOT NULL REFERENCES events(id),
  provider TEXT NOT NULL,
  provider_payment_id TEXT,
  amount NUMERIC(12,2) NOT NULL,
  currency TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE invoices (
  id TEXT PRIMARY KEY,
  payment_id TEXT NOT NULL REFERENCES payments(id),
  invoice_number TEXT NOT NULL UNIQUE,
  pdf_url TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE event_sessions (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  title TEXT NOT NULL,
  description TEXT,
  starts_at TIMESTAMP NOT NULL,
  ends_at TIMESTAMP NOT NULL,
  room_name TEXT,
  sort_order INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE notifications (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  registration_id TEXT REFERENCES registrations(id),
  channel TEXT NOT NULL,
  template_key TEXT NOT NULL,
  status TEXT NOT NULL,
  sent_at TIMESTAMP
);

-- AI bot schema

CREATE TABLE devices (
  id TEXT PRIMARY KEY,
  organization_id TEXT NOT NULL REFERENCES organizations(id),
  device_name TEXT NOT NULL,
  device_type TEXT NOT NULL,
  status TEXT NOT NULL,
  last_seen_at TIMESTAMP
);

CREATE TABLE scan_sessions (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  device_id TEXT NOT NULL REFERENCES devices(id),
  started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  ended_at TIMESTAMP
);

CREATE TABLE checkins (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  registration_id TEXT NOT NULL REFERENCES registrations(id),
  ticket_id TEXT NOT NULL REFERENCES tickets(id),
  checked_in_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  device_id TEXT NOT NULL REFERENCES devices(id),
  method TEXT NOT NULL,
  result TEXT NOT NULL,
  meta_json TEXT
);

CREATE TABLE bot_messages (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  user_id TEXT,
  channel TEXT NOT NULL,
  message_text TEXT NOT NULL,
  intent TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE feedback_surveys (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  title TEXT NOT NULL,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE feedback_responses (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  registration_id TEXT NOT NULL REFERENCES registrations(id),
  survey_id TEXT REFERENCES feedback_surveys(id),
  rating INTEGER,
  feedback_text TEXT,
  would_return BOOLEAN,
  submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE event_incidents (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  severity TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT,
  reported_by TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_logs (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id),
  actor_type TEXT NOT NULL,
  actor_id TEXT,
  action TEXT NOT NULL,
  payload_json TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

