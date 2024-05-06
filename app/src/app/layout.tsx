import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

// import { favicon } from "./icon.ico";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Magic Eraser",
  description:
    "Computer vision project that removes objects with AI Magic Eraser",
  openGraph: {
    // type: "website",
    // url: "https://ma/gic-eraser-mu.vercel.app/",
    title: "Magic Eraser",
    description:
      "Computer vision project that removes objects with AI Magic Eraser",
    // siteName: "Magic Eraser",
    // images: [
    // {
    // url: "/icon",
    // },
    // ],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
