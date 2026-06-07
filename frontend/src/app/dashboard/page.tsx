import Campaigns from "@/src/views/page/campaigns";
import ContentReport from "@/src/views/page/content-report";

export default function DashboardPage() {
  return (
    <>
      <div className="max-w-screen">
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="content-report" className="mt-45 md:mt-15">
            <ContentReport />
          </div>
        </div>
        <div className="px-4 md:px-10 lg:px-16 max-w-8xl mx-auto">
          <div id="campaigns" className="mt-45 md:mt-15">
            <Campaigns />
          </div>
        </div>
      </div>
    </>
  );
}
