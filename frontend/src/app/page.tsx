const highlights = [
  {
    title: "Discovery-first UX",
    text: "Built for browsing, filtering, and joining events quickly."
  },
  {
    title: "Organizer tools",
    text: "Create events, manage tickets, and track attendance from one dashboard."
  },
  {
    title: "Payments ready",
    text: "Stripe-backed ticket checkout and webhook-driven order state."
  },
  {
    title: "Fast check-in",
    text: "QR-based entry with live attendee status updates."
  }
];

export default function HomePage() {
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          Event Platform
        </div>
        <a className="button secondary" href="http://localhost:8000/docs">
          API docs
        </a>
      </header>

      <section className="hero">
        <div className="panel hero-copy">
          <span className="eyebrow">Next.js + FastAPI + PostgreSQL</span>
          <h1>Create, publish, and run events.</h1>
          <p>
            This scaffold gives you the frontend foundation for event discovery,
            organizer workflows, RSVP flows, and ticketing. It is wired to grow
            into the full MVP without needing a rewrite.
          </p>
          <div className="cta-row">
            <button className="button primary" type="button">
              Browse events
            </button>
            <button className="button secondary" type="button">
              Create event
            </button>
          </div>
        </div>

        <div className="panel feature-grid">
          {highlights.map((item) => (
            <article className="feature-card" key={item.title}>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section section-grid">
        <article className="panel section-card">
          <h2>Frontend</h2>
          <ul>
            <li>App router structure</li>
            <li>Landing page shell</li>
            <li>Shared global styles</li>
          </ul>
        </article>
        <article className="panel section-card">
          <h2>Backend</h2>
          <ul>
            <li>FastAPI project structure</li>
            <li>Versioned API routing</li>
            <li>Ready for auth and CRUD</li>
          </ul>
        </article>
        <article className="panel section-card">
          <h2>Infra</h2>
          <ul>
            <li>Docker Compose for Postgres</li>
            <li>Redis for jobs and caching</li>
            <li>Environment template included</li>
          </ul>
        </article>
      </section>
    </main>
  );
}

