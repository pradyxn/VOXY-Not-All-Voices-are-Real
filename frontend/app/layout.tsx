import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "VOXY | Not All Voices Are Real", description: "Voice authenticity detection for safer decisions." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
