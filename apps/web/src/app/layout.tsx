import type { Metadata } from "next";
import { Sidebar, MobileHeader } from "@/components/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Alpha",
  description: "GPT training dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <Sidebar />
        <MobileHeader />
        <main className="min-h-screen lg:pl-56">
          <div className="mx-auto max-w-6xl px-6 py-6">{children}</div>
        </main>
      </body>
    </html>
  );
}
