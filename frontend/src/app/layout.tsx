import type { Metadata } from "next";
import Script from "next/script";
import "../styles/globals.css";
import { ToastContainer } from "react-toastify";

import ReactQueryProvider from "../providers/ReactQueryProvider";
import { ThemeProvider } from "next-themes";
import {
  KEYWORD,
  SITE_DESCRIPTION,
  SITE_TITLE,
  SITE_URL,
  THUMBNAIL,
} from "../constant/metadata";
import Layout from "../layout/Layout";

export const metadata: Metadata = {
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  keywords: KEYWORD,
  robots: {
    follow: true,
    index: true,
    googleBot: {
      index: true,
      follow: true,
    },
  },
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    url: SITE_URL,
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    siteName: SITE_TITLE,
    countryName: "Vietnam",
    images: {
      url: SITE_URL + THUMBNAIL.src,
      secureUrl: THUMBNAIL.src,
      type: "image/png",
      width: THUMBNAIL.width,
      height: THUMBNAIL.height,
    },
  },
  appleWebApp: {
    capable: true,
    title: SITE_TITLE,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" suppressHydrationWarning>
      <head>
        <meta charSet="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />

        <base href="/" />
        <meta
          httpEquiv="Cache-Control"
          content="public, max-age=21600, must-revalidate"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#000000" />
        <meta name="description" content={SITE_DESCRIPTION} />
        <meta itemProp="description" content={SITE_DESCRIPTION} />
        <meta
          name="keywords"
          content="kiểm chứng, AI, tin giả, tài chính, crypto, DeFi, thời gian thực"
        />
        <meta property="og:type" content="website" />
        <meta property="og:title" content={SITE_TITLE} />
        <meta property="og:description" content={SITE_DESCRIPTION} />

        <meta property="og:image" content="/image/logo.png" />
        <meta name="twitter:title" content={SITE_TITLE} />
        <meta name="twitter:description" content={SITE_DESCRIPTION} />
        <meta
          name="robots"
          content="index, follow, max-snippet:-1, max-image-preview:large"
        />
        <meta name="revisit-after" content="0 days" />
        <meta
          name="ROBOTS"
          content="index, follow, max-snippet:-1, max-image-preview:large"
        />
        <meta name="googlebot" content="index,follow" />
        <meta name="BingBOT" content="index,follow" />
        <meta name="yahooBOT" content="index,follow" />
        <meta name="slurp" content="index,follow" />
        <meta name="msnbot" content="index,follow" />
        <meta name="language" content="Vietnamese" />
        <meta property="og:site_name" content={SITE_TITLE} />

        <title>{SITE_TITLE}</title>

        {/* Google Tag Manager */}
        <Script id="google-tag-manager" strategy="afterInteractive">
          {`
          (function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
          new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
          j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
          'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
          })(window,document,'script','dataLayer','GTM-N5CKJ377');
          `}
        </Script>
        {/* End Google Tag Manager */}

        {/* Telegram Widget Script */}
        <Script
          src="https://telegram.org/js/telegram-widget.js?22"
          strategy="afterInteractive"
        />
        {/* End Telegram Widget Script */}
      </head>
      <body
        suppressHydrationWarning
        className="antialiased"
        style={{ fontFamily: "'Inter Tight', 'Tektur', sans-serif" }}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <ReactQueryProvider>
            <Layout>{children}</Layout>
          </ReactQueryProvider>
          {/* {process.env.NODE_ENV === "development" && (
            <div
              style={{
                position: "fixed",
                bottom: 16,
                right: 16,
                cursor: "pointer",
              }}
            >
              <Link href="/assets/typography" title="Typography Test Page">
                <BugIcon />
              </Link>
            </div>
          )} */}
        </ThemeProvider>

        <ToastContainer />
      </body>
    </html>
  );
}
