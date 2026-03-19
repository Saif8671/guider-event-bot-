"use client";

import { useEffect, useMemo, useState } from "react";
import {
  createEvent,
  createRsvp,
  createTicket,
  getMe,
  login,
  publishEvent,
  purchaseTicket,
  signup,
  type AuthUser,
  type EventCreatePayload,
  type EventSummary,
  type TicketSummary
} from "@/lib/api";

type Props = {
  initialEvents: EventSummary[];
};

type AuthMode = "login" | "signup";

type AuthState = {
  mode: AuthMode;
  name: string;
  email: string;
  password: string;
  role: "organizer" | "attendee";
};

type CreateEventState = {
  title: string;
  description: string;
  category: string;
  location_type: string;
  location_name: string;
  location_address: string;
  online_url: string;
  start_at: string;
  end_at: string;
  timezone: string;
  capacity: string;
  is_featured: boolean;
  publish: boolean;
  ticket_name: string;
  ticket_type: string;
  ticket_price_cents: string;
  ticket_quantity_total: string;
};

const defaultAuth: AuthState = {
  mode: "login",
  name: "",
  email: "organizer@example.com",
  password: "password123",
  role: "organizer"
};

function defaultCreateEventState(): CreateEventState {
  const start = new Date();
  start.setDate(start.getDate() + 7);
  start.setHours(18, 0, 0, 0);
  const end = new Date(start);
  end.setHours(end.getHours() + 2);

  return {
    title: "",
    description: "",
    category: "community",
    location_type: "offline",
    location_name: "",
    location_address: "",
    online_url: "",
    start_at: toDatetimeLocal(start),
    end_at: toDatetimeLocal(end),
    timezone: "Asia/Riyadh",
    capacity: "100",
    is_featured: false,
    publish: true,
    ticket_name: "General Admission",
    ticket_type: "free",
    ticket_price_cents: "0",
    ticket_quantity_total: "100"
  };
}

function toDatetimeLocal(value: Date): string {
  const offset = value.getTimezoneOffset();
  const local = new Date(value.getTime() - offset * 60_000);
  return local.toISOString().slice(0, 16);
}

function parseMoney(cents: number, currency: string) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency
  }).format(cents / 100);
}

function mapTickets(tickets: TicketSummary[]): TicketSummary[] {
  return [...tickets].sort((a, b) => a.price_cents - b.price_cents);
}

export default function HomeClient({ initialEvents }: Props) {
  const [events, setEvents] = useState(initialEvents);
  const [authState, setAuthState] = useState<AuthState>(defaultAuth);
  const [createState, setCreateState] = useState<CreateEventState>(() => defaultCreateEventState());
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("Ready.");
  const [busyKey, setBusyKey] = useState<string | null>(null);

  useEffect(() => {
    const storedToken = window.localStorage.getItem("event-platform-token");
    if (!storedToken) {
      return;
    }

    setAccessToken(storedToken);
    getMe(storedToken)
      .then((user) => {
        setCurrentUser(user);
        setStatus(`Signed in as ${user.name}.`);
      })
      .catch(() => {
        window.localStorage.removeItem("event-platform-token");
        setAccessToken(null);
      });
  }, []);

  const featuredCount = useMemo(() => events.filter((event) => event.is_featured).length, [events]);

  async function handleAuthSubmit() {
    setBusyKey("auth");
    try {
      const response =
        authState.mode === "login"
          ? await login(authState.email, authState.password)
          : await signup(authState.name || "New Organizer", authState.email, authState.password, authState.role);
      window.localStorage.setItem("event-platform-token", response.tokens.access_token);
      setAccessToken(response.tokens.access_token);
      setCurrentUser(response.user);
      setStatus(`Authenticated as ${response.user.name}.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Authentication failed.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleCreateEvent() {
    if (!accessToken) {
      setStatus("Log in first to create events.");
      return;
    }

    setBusyKey("create-event");
    try {
      const payload: EventCreatePayload = {
        title: createState.title,
        description: createState.description,
        category: createState.category,
        location_type: createState.location_type,
        location_name: createState.location_name || null,
        location_address: createState.location_address || null,
        online_url: createState.online_url || null,
        start_at: new Date(createState.start_at).toISOString(),
        end_at: new Date(createState.end_at).toISOString(),
        timezone: createState.timezone,
        capacity: Number(createState.capacity),
        is_featured: createState.is_featured
      };

      const event = await createEvent(accessToken, payload);
      const ticket = await createTicket(accessToken, event.id, {
        name: createState.ticket_name,
        ticket_type: createState.ticket_type,
        price_cents: Number(createState.ticket_price_cents),
        quantity_total: Number(createState.ticket_quantity_total)
      });

      const published = createState.publish ? await publishEvent(accessToken, event.id) : event;
      if (createState.publish) {
        const nextEvent: EventSummary = {
          ...published,
          tickets: mapTickets([ticket]),
          created_at: published.created_at,
          updated_at: published.updated_at
        };
        setEvents((current) => [nextEvent, ...current.filter((item) => item.id !== nextEvent.id)]);
      }
      setStatus(`Created ${published.title}${createState.publish ? " and published it" : ""}.`);
      setCreateState(defaultCreateEventState());
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Could not create the event.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handleRsvp(eventId: string, ticketId?: string | null) {
    if (!accessToken) {
      setStatus("Log in first to RSVP.");
      return;
    }

    setBusyKey(`rsvp-${eventId}-${ticketId ?? "free"}`);
    try {
      await createRsvp(accessToken, eventId, { ticket_id: ticketId ?? null });
      setEvents((current) =>
        current.map((event) =>
          event.id === eventId
            ? {
                ...event,
                tickets: event.tickets.map((ticket) =>
                  ticket.id === ticketId ? { ...ticket, quantity_sold: ticket.quantity_sold + 1, remaining_quantity: Math.max(ticket.remaining_quantity - 1, 0) } : ticket
                )
              }
            : event
        )
      );
      setStatus("RSVP confirmed.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "RSVP failed.");
    } finally {
      setBusyKey(null);
    }
  }

  async function handlePurchase(eventId: string, ticketId: string) {
    if (!accessToken) {
      setStatus("Log in first to purchase a ticket.");
      return;
    }

    setBusyKey(`purchase-${eventId}-${ticketId}`);
    try {
      await purchaseTicket(accessToken, eventId, ticketId);
      setEvents((current) =>
        current.map((event) =>
          event.id === eventId
            ? {
                ...event,
                tickets: event.tickets.map((ticket) =>
                  ticket.id === ticketId
                    ? {
                        ...ticket,
                        quantity_sold: ticket.quantity_sold + 1,
                        remaining_quantity: Math.max(ticket.remaining_quantity - 1, 0)
                      }
                    : ticket
                )
              }
            : event
        )
      );
      setStatus("Ticket purchased.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Purchase failed.");
    } finally {
      setBusyKey(null);
    }
  }

  const loginHint = "Use organizer@example.com / password123 or sign up a new organizer.";

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <div className="brand">
            <span className="brand-mark" />
            Event Platform
          </div>
          <p className="topbar-note">Next.js frontend + FastAPI backend</p>
        </div>
        <div className="topbar-actions">
          <a className="button secondary" href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
          <span className="status-badge">{currentUser ? currentUser.email : "Signed out"}</span>
        </div>
      </header>

      <section className="hero">
        <article className="panel hero-copy">
          <span className="eyebrow">Event discovery, registration, and checkout</span>
          <h1>Create events that people can actually join.</h1>
          <p>
            Seeded events are loaded from the backend. Log in, publish a new event, create its first ticket,
            then test the RSVP and purchase flows against the live API.
          </p>
          <div className="hero-metrics">
            <div>
              <strong>{events.length}</strong>
              <span>Published events</span>
            </div>
            <div>
              <strong>{featuredCount}</strong>
              <span>Featured picks</span>
            </div>
            <div>
              <strong>{events.reduce((total, event) => total + event.tickets.length, 0)}</strong>
              <span>Ticket types</span>
            </div>
          </div>
          <p className="status-line">{status}</p>
        </article>

        <aside className="panel auth-card">
          <div className="card-header">
            <div>
              <h2>Sign in</h2>
              <p>{loginHint}</p>
            </div>
            <div className="segmented">
              <button
                className={authState.mode === "login" ? "segmented-active" : ""}
                type="button"
                onClick={() => setAuthState((current) => ({ ...current, mode: "login" }))}
              >
                Login
              </button>
              <button
                className={authState.mode === "signup" ? "segmented-active" : ""}
                type="button"
                onClick={() => setAuthState((current) => ({ ...current, mode: "signup" }))}
              >
                Signup
              </button>
            </div>
          </div>

          <div className="form-grid">
            {authState.mode === "signup" && (
              <label>
                Name
                <input
                  value={authState.name}
                  onChange={(event) => setAuthState((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Organizer name"
                />
              </label>
            )}
            <label>
              Email
              <input
                value={authState.email}
                onChange={(event) => setAuthState((current) => ({ ...current, email: event.target.value }))}
                placeholder="you@example.com"
              />
            </label>
            <label>
              Password
              <input
                type="password"
                value={authState.password}
                onChange={(event) => setAuthState((current) => ({ ...current, password: event.target.value }))}
                placeholder="minimum 8 characters"
              />
            </label>
            {authState.mode === "signup" && (
              <label>
                Role
                <select
                  value={authState.role}
                  onChange={(event) =>
                    setAuthState((current) => ({
                      ...current,
                      role: event.target.value as "organizer" | "attendee"
                    }))
                  }
                >
                  <option value="organizer">Organizer</option>
                  <option value="attendee">Attendee</option>
                </select>
              </label>
            )}
          </div>

          <button className="button primary wide" type="button" onClick={handleAuthSubmit} disabled={busyKey === "auth"}>
            {busyKey === "auth" ? "Working..." : authState.mode === "login" ? "Log in" : "Create account"}
          </button>
        </aside>
      </section>

      <section className="section section-grid">
        <article className="panel section-card create-card">
          <div className="card-header">
            <div>
              <h2>Create event</h2>
              <p>Creates a draft, adds a starter ticket, and can publish immediately.</p>
            </div>
            <span className="status-badge subtle">{currentUser ? currentUser.role : "guest"}</span>
          </div>

          <div className="form-grid two-col">
            <label>
              Title
              <input
                value={createState.title}
                onChange={(event) => setCreateState((current) => ({ ...current, title: event.target.value }))}
                placeholder="Design System Meetup"
              />
            </label>
            <label>
              Category
              <input
                value={createState.category}
                onChange={(event) => setCreateState((current) => ({ ...current, category: event.target.value }))}
                placeholder="community"
              />
            </label>
            <label className="span-2">
              Description
              <textarea
                value={createState.description}
                onChange={(event) => setCreateState((current) => ({ ...current, description: event.target.value }))}
                placeholder="A focused evening session for builders."
                rows={4}
              />
            </label>
            <label>
              Location type
              <select
                value={createState.location_type}
                onChange={(event) => setCreateState((current) => ({ ...current, location_type: event.target.value }))}
              >
                <option value="offline">Offline</option>
                <option value="online">Online</option>
              </select>
            </label>
            <label>
              Location name
              <input
                value={createState.location_name}
                onChange={(event) => setCreateState((current) => ({ ...current, location_name: event.target.value }))}
                placeholder="Venue or platform"
              />
            </label>
            <label>
              Location address
              <input
                value={createState.location_address}
                onChange={(event) => setCreateState((current) => ({ ...current, location_address: event.target.value }))}
                placeholder="Street address"
              />
            </label>
            <label>
              Online URL
              <input
                value={createState.online_url}
                onChange={(event) => setCreateState((current) => ({ ...current, online_url: event.target.value }))}
                placeholder="https://..."
              />
            </label>
            <label>
              Start
              <input
                type="datetime-local"
                value={createState.start_at}
                onChange={(event) => setCreateState((current) => ({ ...current, start_at: event.target.value }))}
              />
            </label>
            <label>
              End
              <input
                type="datetime-local"
                value={createState.end_at}
                onChange={(event) => setCreateState((current) => ({ ...current, end_at: event.target.value }))}
              />
            </label>
            <label>
              Timezone
              <input
                value={createState.timezone}
                onChange={(event) => setCreateState((current) => ({ ...current, timezone: event.target.value }))}
                placeholder="Asia/Riyadh"
              />
            </label>
            <label>
              Capacity
              <input
                type="number"
                min="0"
                value={createState.capacity}
                onChange={(event) => setCreateState((current) => ({ ...current, capacity: event.target.value }))}
              />
            </label>
            <label>
              Ticket name
              <input
                value={createState.ticket_name}
                onChange={(event) => setCreateState((current) => ({ ...current, ticket_name: event.target.value }))}
                placeholder="General admission"
              />
            </label>
            <label>
              Ticket type
              <select
                value={createState.ticket_type}
                onChange={(event) => setCreateState((current) => ({ ...current, ticket_type: event.target.value }))}
              >
                <option value="free">Free</option>
                <option value="paid">Paid</option>
              </select>
            </label>
            <label>
              Price in cents
              <input
                type="number"
                min="0"
                value={createState.ticket_price_cents}
                onChange={(event) => setCreateState((current) => ({ ...current, ticket_price_cents: event.target.value }))}
              />
            </label>
            <label>
              Ticket quantity
              <input
                type="number"
                min="1"
                value={createState.ticket_quantity_total}
                onChange={(event) => setCreateState((current) => ({ ...current, ticket_quantity_total: event.target.value }))}
              />
            </label>
          </div>

          <div className="inline-options">
            <label>
              <input
                type="checkbox"
                checked={createState.is_featured}
                onChange={(event) => setCreateState((current) => ({ ...current, is_featured: event.target.checked }))}
              />
              Featured
            </label>
            <label>
              <input
                type="checkbox"
                checked={createState.publish}
                onChange={(event) => setCreateState((current) => ({ ...current, publish: event.target.checked }))}
              />
              Publish immediately
            </label>
          </div>

          <button
            className="button primary wide"
            type="button"
            onClick={handleCreateEvent}
            disabled={busyKey === "create-event"}
          >
            {busyKey === "create-event" ? "Creating..." : "Create event"}
          </button>
        </article>

        <article className="panel section-card intro-card">
          <h2>Seeded access</h2>
          <p>
            The backend seed script creates <strong>organizer@example.com</strong> and{" "}
            <strong>attendee@example.com</strong> with password <strong>password123</strong>.
          </p>
          <p>
            Use the login form, then RSVP to free tickets or purchase paid ones directly from the event cards.
          </p>
          <div className="event-stats">
            <div>
              <strong>{currentUser ? currentUser.name : "Guest"}</strong>
              <span>Session</span>
            </div>
            <div>
              <strong>{events.filter((event) => event.tickets.some((ticket) => ticket.price_cents === 0)).length}</strong>
              <span>Free events</span>
            </div>
          </div>
        </article>
      </section>

      <section className="section">
        <div className="section-heading">
          <div>
            <span className="eyebrow">Discovery feed</span>
            <h2>Upcoming events</h2>
          </div>
          <p>
            {events.length} public events loaded from the API. Feature cards expose both RSVP and paid ticket flows.
          </p>
        </div>

        <div className="event-list-grid">
          {events.map((event) => {
            const sortedTickets = mapTickets(event.tickets);
            const freeTicket = sortedTickets.find((ticket) => ticket.price_cents === 0);
            return (
              <article className="panel event-card" key={event.id}>
                <div className="event-card-top">
                  <div>
                    <div className="event-labels">
                      <span className={`status-pill ${event.is_featured ? "featured" : ""}`}>
                        {event.is_featured ? "Featured" : "Public"}
                      </span>
                      <span className="status-pill muted">{event.category}</span>
                    </div>
                    <h3>{event.title}</h3>
                    <p>{event.description}</p>
                  </div>
                </div>

                <div className="event-meta">
                  <span>{new Date(event.start_at).toLocaleString()}</span>
                  <span>{event.location_type === "online" ? event.online_url || "Online" : event.location_name || "Offline"}</span>
                  <span>Capacity {event.capacity}</span>
                </div>

                <div className="ticket-list">
                  {sortedTickets.map((ticket) => (
                    <div className="ticket-row" key={ticket.id}>
                      <div>
                        <strong>{ticket.name}</strong>
                        <p>
                          {ticket.ticket_type} - {ticket.remaining_quantity} left
                        </p>
                      </div>
                      <div className="ticket-actions">
                        <span>{parseMoney(ticket.price_cents, ticket.currency)}</span>
                        {ticket.price_cents === 0 ? (
                          <button
                            className="button secondary"
                            type="button"
                            onClick={() => handleRsvp(event.id, ticket.id)}
                            disabled={busyKey === `rsvp-${event.id}-${ticket.id}`}
                          >
                            {busyKey === `rsvp-${event.id}-${ticket.id}` ? "Submitting..." : "RSVP"}
                          </button>
                        ) : (
                          <button
                            className="button secondary"
                            type="button"
                            onClick={() => handlePurchase(event.id, ticket.id)}
                            disabled={busyKey === `purchase-${event.id}-${ticket.id}`}
                          >
                            {busyKey === `purchase-${event.id}-${ticket.id}` ? "Purchasing..." : "Buy ticket"}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {freeTicket ? (
                  <div className="card-footer">
                    <button
                      className="button primary"
                      type="button"
                      onClick={() => handleRsvp(event.id, freeTicket.id)}
                      disabled={busyKey === `rsvp-${event.id}-${freeTicket.id}`}
                    >
                      RSVP to free ticket
                    </button>
                    <span>Fast path for attendee signup.</span>
                  </div>
                ) : null}
              </article>
            );
          })}
        </div>
      </section>
    </main>
  );
}
