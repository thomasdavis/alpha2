import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: ["@alpha/core", "@alpha/db"],
  async rewrites() {
    return [
      { source: "/inference", destination: "/inference.html" },
      { source: "/chat", destination: "/chat.html" },
      { source: "/docs", destination: "/docs.html" },
      { source: "/models", destination: "/models.html" },
    ];
  },
};

export default nextConfig;
