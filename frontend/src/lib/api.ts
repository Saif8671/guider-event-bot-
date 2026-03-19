export type EventSummary = {
  id: string;
  title: string;
  slug: string;
  description: string;
  category: string;
  location_type: string;
  location_name: string | null;
  online_url: string | null;
  start_at: string;
  end_at: string;
  timezone: string;
  cover_image_url: string | null;
  status: string;
  capacity: number;
  is_featured: boolean;
};

export type EventListResponse = {
  items: EventSummary[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

export async function getPublishedEvents(): Promise<EventSummary[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/events?limit=6`, {
      cache: "no-store"
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as EventListResponse;
    return data.items ?? [];
  } catch {
    return [];
  }
}

