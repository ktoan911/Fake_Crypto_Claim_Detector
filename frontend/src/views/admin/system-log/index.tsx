import IntroAppPage from "@/src/components/IntroAppPage";
import SystemLogPanel from "./SystemLog";

export default function SystemLogSection() {
  return (
    <>
      <IntroAppPage
        title="System Logs"
        description="Raw server output - tail -f /var/log/crawler/system.log"
      ></IntroAppPage>
      <SystemLogPanel />
    </>
  );
}
