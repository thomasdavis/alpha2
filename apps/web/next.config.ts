import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: [
    "@alpha/core", "@alpha/db", "@alpha/tensor", "@alpha/autograd",
    "@alpha/tokenizers", "@alpha/model", "@alpha/train",
  ],
  serverExternalPackages: [],
  async rewrites() {
    return [
      { source: "/chat/completions", destination: "/v1/chat/completions" },
    ];
  },
  experimental: {
    serverActions: {
      bodySizeLimit: "50mb",
    },
  },
};

export default nextConfig;
