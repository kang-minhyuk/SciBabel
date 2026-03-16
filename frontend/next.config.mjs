/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Reduce exposure to image optimizer DoS advisories on self-hosted deployments.
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
