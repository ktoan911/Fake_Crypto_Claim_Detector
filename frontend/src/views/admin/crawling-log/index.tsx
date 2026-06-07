import IntroAppPage from "@/src/components/IntroAppPage";
import CrawlingLogPanel from "./CrawlingLog";

export default function CrawlingLogSection() {
  return (
    <>
      <IntroAppPage
        title="Crawling Logs"
        description="Live event stream from all active crawlers and data sources"
      />
      <CrawlingLogPanel />
    </>
  );
}
