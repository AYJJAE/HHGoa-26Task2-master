import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    VITE_API_BASE_URL: process.env.VITE_API_BASE_URL || process.env.NEXT_PUBLIC_API_URL || "",
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || process.env.VITE_API_BASE_URL || "",
  },
};

export default nextConfig;
