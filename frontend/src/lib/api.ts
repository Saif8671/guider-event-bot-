export type AuthUser = {
  id: string;
  name: string;
  email: string;
  role: string;
};

export type AuthTokens = {
  access_token: string;
  refresh_token: string;
  token_type: string;
};

export type AuthResponse = {
  user: AuthUser;
  tokens: AuthTokens;
};

export type TicketSummary = {
  id: string;
  event_id: string;
  name: string;
  ticket_type: string;
  price_cents: number;
  currency: string;
  quantity_total: number;
  quantity_sold: number;
  remaining_quantity: number;
  sale_starts_at: string | null;
  sale_ends_at: string | null;
  requires_approval: boolean;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
};

export type EventSummary = {
  id: string;
  organizer_id: string;
  title: string;
  slug: string;
  description: string;
  category: string;
  location_type: string;
  location_name: string | null;
  location_address: string | null;
  online_url: string | null;
  start_at: string;
  end_at: string;
  timezone: string;
  cover_image_url: string | null;
  status: string;
  capacity: number;
  is_featured: boolean;
  tickets: TicketSummary[];
  created_at: string | null;
  updated_at: string | null;
};

export type EventListResponse = {
  items: EventSummary[];
};

export type EventCreatePayload = {
  title: string;
  description?: string;
  category?: string;
  location_type?: string;
  location_name?: string | null;
  location_address?: string | null;
  online_url?: string | null;
  start_at: string;
  end_at: string;
  timezone: string;
  cover_image_url?: string | null;
  capacity?: number;
  is_featured?: boolean;
};

export type TicketCreatePayload = {
  name: string;
  ticket_type?: string;
  price_cents?: number;
  currency?: string;
  quantity_total?: number;
  sale_starts_at?: string | null;
  sale_ends_at?: string | null;
  requires_approval?: boolean;
  is_active?: boolean;
};

export type RSVPResponse = {
  id: string;
  event_id: string;
  user_id: string;
  ticket_id: string | null;
  status: string;
  qr_code_token: string;
  checked_in_at: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type OrderResponse = {
  id: string;
  user_id: string;
  event_id: string;
  ticket_id: string | null;
  payment_provider: string;
  provider_payment_id: string | null;
  amount_cents: number;
  currency: string;
  status: string;
  refunded_at: string | null;
  created_at: string | null;
  updated_at: string | null;
};

export type PurchaseResponse = {
  order: OrderResponse;
  rsvp: RSVPResponse | null;
};

export type RSVPCreatePayload = {
  ticket_id?: string | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
  accessToken?: string
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store"
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function getPublishedEvents(): Promise<EventSummary[]> {
  try {
    const data = await apiFetch<EventListResponse>("/events?limit=12");
    return data.items ?? [];
  } catch {
    return [];
  }
}

export async function login(email: string, password: string): Promise<AuthResponse> {
  return apiFetch<AuthResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password })
  });
}

export async function signup(
  name: string,
  email: string,
  password: string,
  role: "organizer" | "attendee"
): Promise<AuthResponse> {
  return apiFetch<AuthResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ name, email, password, role })
  });
}

export async function getMe(accessToken: string): Promise<AuthUser> {
  const data = await apiFetch<{ user: AuthUser }>("/auth/me", {}, accessToken);
  return data.user;
}

export async function createEvent(accessToken: string, payload: EventCreatePayload): Promise<EventSummary> {
  return apiFetch<EventSummary>("/events", {
    method: "POST",
    body: JSON.stringify(payload)
  }, accessToken);
}

export async function publishEvent(accessToken: string, eventId: string): Promise<EventSummary> {
  return apiFetch<EventSummary>(`/events/${eventId}/publish`, { method: "POST" }, accessToken);
}

export async function createTicket(
  accessToken: string,
  eventId: string,
  payload: TicketCreatePayload
): Promise<TicketSummary> {
  return apiFetch<TicketSummary>(`/events/${eventId}/tickets`, {
    method: "POST",
    body: JSON.stringify(payload)
  }, accessToken);
}

export async function createRsvp(
  accessToken: string,
  eventId: string,
  payload: RSVPCreatePayload
): Promise<RSVPResponse> {
  return apiFetch<RSVPResponse>(`/events/${eventId}/rsvps`, {
    method: "POST",
    body: JSON.stringify(payload)
  }, accessToken);
}

export async function purchaseTicket(
  accessToken: string,
  eventId: string,
  ticketId: string
): Promise<PurchaseResponse> {
  return apiFetch<PurchaseResponse>(`/events/${eventId}/tickets/${ticketId}/purchase`, {
    method: "POST"
  }, accessToken);
}
