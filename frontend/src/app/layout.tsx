import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Event Platform",
  description: "Next.js frontend for the event platform MVP."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

