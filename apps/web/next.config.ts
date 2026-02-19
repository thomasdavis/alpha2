import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: ["@alpha/core", "@alpha/db"],
};

export default nextConfig;
