import type { Metadata } from "next";
import { Sidebar, MobileHeader } from "@/components/sidebar";
import { Providers } from "./providers";
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
    <html lang="en" suppressHydrationWarning>
      <body className="font-sans antialiased">
        <Providers>
          <Sidebar />
          <MobileHeader />
          <main className="min-h-screen lg:pl-56">
            <div className="mx-auto max-w-6xl px-6 py-6">{children}</div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
