import type { Metadata } from "next";
import { Sidebar, MobileHeader } from "@/components/sidebar";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Alpha",
  description: "Public research workbench for Alpha models and synthetic conversational data",
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
            <div className="mx-auto max-w-[112rem] px-4 py-5 sm:px-6 sm:py-6">{children}</div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
