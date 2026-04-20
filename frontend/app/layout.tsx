import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Synkora AI — Document Intelligence",
  description:
    "Ask questions across your uploaded PDFs with grounded, citation-backed answers.",
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%231a73e8'/><text y='22' x='6' font-size='18' fill='white' font-family='sans-serif' font-weight='700'>S</text></svg>",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600&family=Google+Sans+Display:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
