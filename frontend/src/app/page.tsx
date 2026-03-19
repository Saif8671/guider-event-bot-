import { getPublishedEvents } from "@/lib/api";
import HomeClient from "./home-client";

export default async function HomePage() {
  const events = await getPublishedEvents();
  return <HomeClient initialEvents={events} />;
}
