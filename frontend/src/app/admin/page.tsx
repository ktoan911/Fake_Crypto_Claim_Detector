import IntroAppPage from "@/src/components/IntroAppPage";
import DataAnalyst from "@/src/views/admin/data-analyst";
import SystemLogSection from "@/src/views/admin/system-log";
import CrawlingLogSection from "@/src/views/admin/crawling-log";
import ContentTotal from "@/src/views/page/content-report/ContentTotal";

export default function page() {
  return (
    <>
      <div className="max-w-screen">
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="content-report" className="mt-45 md:mt-15">
            <IntroAppPage
              title="Báo Cáo Nội Dung"
              description="Thống kê tin tức tài chính / ngân hàng được phát hiện theo ngày và theo nhóm."
            />
            <div className="mt-10">
              <ContentTotal />
            </div>
          </div>
        </div>
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="campaigns" className="mt-45 md:mt-15">
            <DataAnalyst />
          </div>
        </div>
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="crawling-logs" className="mt-45 md:mt-15">
            <CrawlingLogSection />
          </div>
        </div>
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="server-logs" className="mt-45 md:mt-15">
            <SystemLogSection />
          </div>
        </div>
      </div>
    </>
  );
}
