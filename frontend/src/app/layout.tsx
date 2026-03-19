import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Event Platform",
  description: "Event discovery, login, creation, RSVP, and ticket purchase flows."
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
